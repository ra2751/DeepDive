# DeepDive AI architecture

This document covers what the [README](../README.md) does not: module layout, shared state, the
verification pipeline in detail, budgets, and how evaluation is recorded.

## Module layout

| Package | Role |
|---|---|
| `app.py`, `ui/` | Streamlit UI: sidebar controls, four result tabs, follow-up, Markdown export (`ui/research_view.py`) |
| `api/` | FastAPI service (`GET /health`) |
| `graph_builder/research_graph.py` | The nine-node LangGraph workflow, routers and `run_research()` |
| `agents/` | One module per node (`intake_agent.py` … `report_agent.py`), shared helpers in `research_common.py`, state in `research_state.py` |
| `verification/` | Deterministic rules: quote and relevance checks, claim-status aggregation, sentence grounding, negation and forecast detection, source quality, meta-claim filter |
| `gateway/mesh.py` | The only place model clients are created; every chat, structured-output and embedding call goes through Mesh |
| `budget/` | Per-run `BudgetManager` (limits, usage ledger, cost estimates) and the Mesh price table |
| `retriever/` | Milvus document search (`retriever_pipeline.py`) and the Tavily web-search adapter (`web_search.py`) |
| `pdf_processing/`, `vector_database/`, `main_pipeline.py` | PDF upload and ingestion into Milvus (text, tables, images), within the ingestion budget |
| `domains/configs/` | Domain policies (scope, claim types, high-risk types, source requirements) |
| `config/` | Settings loader, Mesh price table, source-quality rules |
| `evals/` | Live-evaluation harnesses, datasets and recorded results |
| `tests/` | Offline test suite (network blocked; scripted Mesh, Milvus and Tavily fakes) |

The sidebar also offers an **answer-level** workflow (`graph_builder/graph_builder.py` and the `agents/*_node.py`
modules). It checks answers as a whole rather than claim by claim, and never labels findings as verified.

## Shared state

`agents/research_state.ResearchState` is a typed dictionary passed between nodes. Evidence, claims and
verifications are keyed by ID and merged across rounds. `trace` and `notices` are append-only, so every
hand-off and warning is preserved for the Audit tab. Evidence IDs are deterministic hashes of source, page and
text, so the same passage keeps the same ID across rounds and follow-ups.

## Verification pipeline

1. **Judgment** (`verifier_agent.py`): the verifier model receives each claim with the IDs of the passages
   shown for it and returns a stance plus a verbatim quote per pair. Claims are batched (5 per call).
2. **Acceptance** (`verification/aggregation.py`, `citation_integrity.py`): a judgment counts only if the
   evidence exists, was shown for that claim and the quote is verbatim. For support, the quote must also:
   - contain every claim number;
   - not change the claim's negation;
   - cover enough of its key terms. A terse quote can instead pass a narrow number-anchored rule: the same
     number and unit, with the subject covered by the passage.
3. **Status:** computed from accepted judgments and the domain's independent-source requirement (distinct
   documents or web hosts). Model confidence is never used.
4. **Retry** (`retry_agent.py`): insufficient claims are searched again by their own text; only new passages
   are added, and re-verification sees them.
5. **Grounding** (`verification/grounding.py`): each synthesized sentence must cite supported claims, add no
   numbers, and add no ranking, counting, causal, category, negation or forecast-to-fact wording absent from
   those claims. Otherwise every uncovered supported claim is shown as its verified text. This check rejects
   visible additions; it does not prove entailment.
6. **Citations:** built only from accepted evidence and validated again (source, page or URL, verbatim quote).

Source-quality tiers (`config/source_quality.yaml`) are matched on exact host, then the longest parent host,
then official suffixes (`.gov`, `.gov.in`, `.nic.in`, …). They order evidence and are displayed, but never
change a status.

## Budgets

Every run gets a `BudgetManager` passed through LangGraph's config. The gateway and the web adapter check it
before **each attempt**, including retries, and record:
- usage: tokens, reasoning tokens, cached responses and latency;
- the cost basis: reported, estimated or unknown.

When a limit is reached, the run stops gracefully. Unverified claims become unresolved, and the report is
marked partial. Ingestion has its own budget.

## Follow-up

`prior_context_from()` passes the previous question, its verified findings and up to 20 accepted evidence
passages to the next run in the same browser session. Reused evidence is filtered to the current source scope
and domain, labelled "reused", and verified again. Nothing inherits a previous status.

## Evaluation records

`evals/run_live_eval.py` (labelled cases) and `evals/ui_pilot.py` (Streamlit runs) work as follows:
- Each runs only with `--confirm-live`.
- Each enforces per-case request caps through persistent ledgers, so a case cannot be re-run silently.
- Each writes results with the per-call telemetry.

`evals/summarize_results.py` builds `evals/results/SUMMARY.md` from recorded files only.
