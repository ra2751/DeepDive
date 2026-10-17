# Evaluation results

Live evaluation results are written here by `python -m evals.run_live_eval --confirm-live`, in the
schema defined by [`evals/result_schema.py`](../result_schema.py) (`schema_version` 1.0). Each run produces:

- `<UTC timestamp>_<commit>.json`: the full `EvalRun` record. It includes the dataset SHA-256, the Mesh
  models requested and served, the limits, each case's expected and observed outcome, and request counts.
  Costs are labelled `reported`, `estimated` or `unknown`.
- `<UTC timestamp>_<commit>.md`: a human-readable summary.

`kind` is `live` for real Mesh runs. `offline_fake` runs (scripted models) only check the pipeline;
they are never reported as model accuracy.

**Status: no live evaluation has been run yet.**
