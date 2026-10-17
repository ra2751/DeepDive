# DeepDive evaluation sets

| Dataset | What it tests | How it runs | Status |
|---|---|---|---|
| `datasets/verification_rules_v1.jsonl` | The deterministic verification layer: quote integrity, relevance, status rules, source independence, domain policies | Offline, `pytest tests/test_eval_rules_dataset.py` (no model calls) | Runs in CI |
| `datasets/live_eval_v1.jsonl` | The full workflow with real Mesh models against the existing Milvus index (4 public papers) | Live only, after approval (`python -m evals.run_live_eval --confirm-live`) | Run: results in `results/`, summarised in `results/SUMMARY.md` |

**Do not read the offline results as model accuracy.** The rules set feeds hand-written judgments into the deterministic code. It shows that the rules behave as specified, not how well any model verifies claims.

## Labels
- **Rules set:** expected status and expected rejection reasons for each synthetic case. All fixture documents are synthetic.
- **Live set:** expected outcomes are taken from the published papers ("Attention Is All You Need", "A Survey of Transformers"). Confirm that the indexed chunks contain each passage before scoring. Several cases accept more than one status (e.g. `contradicted` or `unresolved` for false-claim probes).

## Metrics (live set)
- **Claim-status agreement:** the share of fact-check claims whose status is in the accepted set.
- **Unsupported-claim leakage:** report sentences citing a claim that is not `supported`. Target 0; the citation check enforces it.
- **Citation integrity issues:** target 0.
- **Requests and cost:** the Budget Manager's counts, with each cost labelled reported, estimated or unknown.

## Recorded results (`results/`)

| Files | Run | Outcome |
|---|---|---|
| `20261017T*_9a16a91.*`, `SUMMARY.md` | Labelled live set: 10 questions, 11 observations (L08-r1 is a rerun) | 8/10 first attempt, 9/10 latest |
| `20261017_B4_ui_pilot_*` | Two questions through the Streamlit UI (research + follow-up) | Q1 pass, Q2 partial |
| `20261017_GR1_web_only_*` | One open-topic Web-only research question (`python -m evals.ui_pilot --pilot gr1`) | Partial (see the main README) |

Recorded files are never edited after a run; fixes are re-measured as new runs.
