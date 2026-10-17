"""Synthesizer: final claim statuses, then report sentences from SUPPORTED claims only.

Unsupported claims are never sent to the synthesizer. If no model call is possible, a
deterministic template (one sentence per supported claim) is used instead.
"""

from langchain_core.runnables import RunnableConfig

from agents.research_common import UNTRUSTED_NOTE, step
from budget import BudgetExceeded, budget_from_config
from gateway import MeshError, structured
from schemas.research import ClaimVerification, ReportSentence, SynthesisOutput
from verification.aggregation import finalize_status
from verification.temporal import is_forecast

PROMPT = """Write a concise research answer for a {domain_name} audience.
{untrusted}

<question>{question}</question>

Use ONLY these verified claims. Every sentence must list the IDs of the claims it uses in claim_ids,
and must not add any fact, number, date or name that is not in those claims. Restate the claims; do not
interpret them:
- no rankings or importance words ("biggest", "leading", "key", "most important") unless the claim says so;
- no counts of findings ("three opportunities") and never imply that every requested item was found;
- no causes, consequences, conditions, implications or risks that the claim does not state;
- do not call a fact a risk, opportunity or challenge unless the claim itself does;
- do not merge claims into a new conclusion; one sentence per claim is fine.
Sentences that add anything are replaced by the plain claim text. Claims marked [forecast] are
projections reported by a source: present them as projections, never as observed results. Do NOT write claim IDs
such as (C1) or [C1] inside the sentence text; they are added automatically:
{claims}"""


def SynthesizerAgent(state, config: RunnableConfig):
    budget = budget_from_config(config)
    claims, verifications = state.get("claims") or {}, dict(state.get("verifications") or {})
    reason = state.get("termination_reason")
    final_reason = f"run stopped ({reason})" if reason else "still insufficient after the final retry round"
    finalized = {cid: finalize_status(v, final_reason) for cid, v in verifications.items()}
    for cid in claims:  # claims that were never verified (e.g. budget stop) are unresolved, not supported
        if cid not in finalized:
            finalized[cid] = ClaimVerification(claim_id=cid, status="unresolved", rule=f"not verified; {final_reason}")

    supported = [cid for cid, v in finalized.items() if v.status == "supported"]
    mode, sentences = "none", []
    if supported:
        template = [ReportSentence(text=claims[c].text, claim_ids=[c], origin="template") for c in supported]
        if state.get("claims_to_check"):
            # Fact-check mode: the verified user claims ARE the findings; no extra Mesh call.
            mode, sentences = "template (fact-check)", template
        elif reason or (budget.llm_calls_remaining() == 0):
            mode, sentences = "template", template
        else:
            prompt = PROMPT.format(domain_name=state["domain"].name, untrusted=UNTRUSTED_NOTE,
                                   question=(state.get("plan").resolved_question if state.get("plan") else "")
                                   or state["query"],
                                   claims="\n".join(f"{c}{' [forecast]' if is_forecast(claims[c].text, claims[c].claim_type) else ''}: "
                                                     f"{claims[c].text}" for c in supported))
            try:
                out: SynthesisOutput = structured("generator", SynthesisOutput, budget).invoke(prompt)
                mode, sentences = "model", out.sentences
            except (BudgetExceeded, MeshError) as e:
                mode, sentences = "template", template
                reason = reason or getattr(e, "reason", None)
    out = {"verifications": finalized, "draft_sentences": sentences,
           "trace": [step("synthesize", agent="Synthesizer", supported_claims=len(supported), mode=mode,
                          sentences=len(sentences))]}
    if reason and not state.get("termination_reason"):
        out["termination_reason"] = reason
    return out
