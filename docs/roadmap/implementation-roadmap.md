# DeepDive AI — Implementation Roadmap

| | |
|---|---|
| **Status** | Proposed — awaiting approval before any code change |
| **Date** | 2026-10-16 |
| **Related** | [Current assessment](../architecture/architecture-assessment.md) · [Target architecture](../architecture/target-architecture.md) · [Technical decisions](../architecture/technical-decisions.md) |

The roadmap is ordered by the hackathon priorities:

1. end-to-end workflow
2. Mesh compliance
3. claim verification and citations
4. bounded self-correction
5. usable UI
6. document and web research
7. local persistence
8. tests and evaluation
9. basic multimodal, if time allows

Each phase leaves the application runnable. Sizes are relative effort (S/M/L), not calendar estimates.

**Key to file actions:** **New** = file created · **Modify** = existing file changed · **Remove** = deleted after its replacement passes tests.

---

## 1. MVP scope

### In scope (Agent Arena MVP)

- Mesh API gateway; **all** runtime inference through Mesh
- Verified research graph: intake → plan → research → extract → verify ⇄ targeted retry → synthesize → citation check → finalize
- Evidence-based claim statuses (`supported`, `contradicted`, `insufficient_evidence`, `unresolved`) with exact-quote checks
- Configurable bounds: retry rounds, LLM and tool calls, timeouts, cost
- Domain configs: general, procurement, supply chain, pharma, insurance
- PDF upload through a `LocalUploadConnector` behind the connector interface; text and tables indexed; web search
- Milvus v2 collection with document, version, tenant and ACL fields
- SQLite operational database (SQLAlchemy + Alembic) and SQLite LangGraph checkpointer
- Streamlit: research, progress, report, claims, evidence, audit, history, download
- pytest suite, Mesh compliance test, small evaluation set
- Dockerfile for local or container runs

### Out of scope for the MVP

- SharePoint, OneDrive, Google Drive, Confluence, S3, GCS and Azure Blob connectors; incremental sync; webhooks
- Enterprise SSO, multi-tenant administration, row-level security
- OCR and vision interpretation (unless Phase 6 time allows and OQ-02 is resolved)
- Cloud SQL, GCS and Cloud Run production deployment
- Human-in-the-loop approval gates (MVP shows review flags only)
- Hash-chained audit log, `tenants`, `connector_configs` and `evaluation_results` tables

---

## 2. Phase overview

| Phase | Name | Size | Depends on | Outcome |
|---|---|---|---|---|
| 0 | Architecture assessment | — | — | **Done**: these documents |
| 1 | Mesh-compliant, bounded foundation | S | Mesh key, OQ-01 | Existing app runs entirely via Mesh; loop bug fixed; test scaffold |
| 2 | Verified research graph | L | 1 | New workflow end-to-end with claims, verification and bounded retry |
| 3 | Persistence and ingestion abstraction | M | 2 | SQLite system of record, checkpointer, connector interface, Milvus v2 |
| 4 | Streamlit experience | M | 2, 3 | Multipage UI with progress, claims, evidence, audit, history, download |
| 5 | Evaluation, hardening, container | M | 2–4 | Eval harness, structured logs, Dockerfile, cleanup |
| 6 | Basic multimodal (stretch) | S–M | 3, OQ-02 | Tables are already indexed; image captions via Mesh vision |
| 7+ | Post-MVP enterprise | L | — | Postgres/GCP, connectors, identity, tenancy, review gates |

---

## 3. Phase 1 — Mesh-compliant, bounded foundation (first milestone)

**Goal:** keep current behaviour, but route every inference call through Mesh, make the retry loop genuinely bounded, remove work done at import time, and add a test suite.

| Action | Path | Change |
|---|---|---|
| New | `config/settings.py` | Pydantic settings: `MESH_API_KEY`, `MESH_BASE_URL`, per-role model IDs, embedding model, Milvus, Tavily, budget defaults. Missing required values fail with a clear message |
| Modify | [config/config.py](../../config/config.py) | Thin compatibility layer that delegates to the gateway; remove the `os.environ` writes and the broken `get_llm_validation` |
| New | `gateway/__init__.py`, `gateway/mesh.py` | `chat(role)`, `structured(role, schema)`, `embeddings()`, `probe_capabilities()`, usage callback |
| Modify | [pdf_processing/embeddings.py](../../pdf_processing/embeddings.py) | Use `gateway.embeddings()` |
| Modify | [agents/validation_node.py](../../agents/validation_node.py) | Remove the Gemini import and the client created at import; use `gateway.structured("verifier", ValidationOutput)` |
| Modify | [agents/supervisor_node.py](../../agents/supervisor_node.py), [rag_node.py](../../agents/rag_node.py), [web_node.py](../../agents/web_node.py), [llm_node.py](../../agents/llm_node.py) | Get models from the gateway; stop logging full document content |
| Modify | [agents/validation_router.py](../../agents/validation_router.py) | Fix `"end_note"`; route to finalize when `loop_count` has reached the maximum, whatever the verdict |
| Modify | [agents/state_node.py](../../agents/state_node.py) | Type `validation_metadata` as `dict` |
| Modify | [graph_builder/graph_builder.py](../../graph_builder/graph_builder.py) | `build_graph(checkpointer=None)` factory; no compile at import |
| Modify | [app.py](../../app.py) | Call `build_graph()` once (cached with `st.cache_resource`); pass an explicit `recursion_limit` |
| Modify | [retriever/retriever_pipeline.py](../../retriever/retriever_pipeline.py) | Use gateway embeddings; guard empty results before MMR |
| Modify | [requirements.txt](../../requirements.txt) | Remove direct-provider and unused packages; add `pytest`; pin versions |
| New | `.env.example` | Variable names only, no values |
| New | `pytest.ini` (or `[tool.pytest]` section), `tests/conftest.py` | Fake gateway fixture (`GenericFakeChatModel`, fake embeddings); `live` marker |
| New | `tests/test_mesh_compliance.py` | Fails if a direct provider import or client construction appears outside `gateway/` |
| New | `tests/test_graph_bounds.py` | Validator always says "no" → graph ends within the maximum passes, with no exception |
| New | `tests/integration/test_mesh_live.py` | `@live`: chat, structured output and 1536-dimension embeddings via Mesh |

**Acceptance criteria:**

1. No module outside `gateway/` imports `openai`, `langchain_google_genai` or `langchain_anthropic`, or constructs `ChatOpenAI`, `OpenAIEmbeddings` or `init_chat_model`. Enforced by `tests/test_mesh_compliance.py`.
2. With only `MESH_API_KEY`, `TAVILY_API_KEY` and the Milvus variables set (no `OPENAI_API_KEY` or `GOOGLE_API_KEY`), one RAG question and one web question complete end-to-end in the Streamlit app.
3. Mesh embeddings return 1536-dimension vectors (live test).
4. The validator returns a parsed `ValidationOutput` via Mesh. When structured output is unsupported, the fallback path runs and is logged.
5. If the validator returns "no" on every pass, the graph ends in ≤3 supervisor passes with no `GraphRecursionError` and no unknown-branch error.
6. Importing any project module makes no network calls and needs no API keys.
7. `pytest -m "not live"` passes offline in under 30 seconds.
8. Each model call logs the model requested, the model served, tokens and latency, and no document content.

---

## 4. Phase 2 — Verified research graph

**Goal:** replace answer-level routing and judging with the claim-level workflow described in [target-architecture.md §3 and §5](../architecture/target-architecture.md#3-research-workflow-langgraph).

| Action | Path | Change |
|---|---|---|
| New | `schemas/__init__.py`, `schemas/research.py` | `ResearchPlan`, `SubQuestion`, `ReportDraft`, `ProgressEvent`, `Budget`, `Usage` |
| New | `schemas/evidence.py` | `SourceLocator`, `Evidence`, `Claim`, `EvidenceJudgment`, `ClaimVerification` |
| New | `schemas/domain.py` | `DomainConfig` and its verification, web and report sub-models |
| New | `domains/__init__.py`, `domains/registry.py` | Load and validate YAML configs; record each config's version |
| New | `domains/configs/{general,procurement,supply_chain,pharma,insurance}.yaml` | Initial domain configs |
| Modify | [agents/state_node.py](../../agents/state_node.py) | Add `ResearchState` with reducers; keep `AgentRagState` until the old graph is removed |
| New | `agents/intake_node.py`, `agents/planner_node.py` | Domain selection and scope check; bounded plan |
| New | `agents/research_node.py`, `agents/targeted_research_node.py` | Evidence collection per sub-question and per insufficient claim |
| New | `agents/claim_extraction_node.py`, `agents/verification_node.py` | Atomic claims; independent retrieval; per-pair judgments |
| New | `agents/synthesis_node.py`, `agents/citation_check_node.py` | Report written from supported claims; marker integrity |
| Modify | [agents/final_response_node.py](../../agents/final_response_node.py) | Becomes `finalize`: assemble the final report and status |
| New | `agents/routers.py` | `after_intake`, `after_verify`, `after_citation_check`, `budget_exhausted` |
| New | `retriever/document_search.py` | Wraps `RetrieverPipeline`; returns `Evidence` with page and chunk locators |
| New | `retriever/web_search.py` | Tavily search extracted from [web_node.py](../../agents/web_node.py); domain allow and block lists; `retrieved_at` |
| New | `verification/__init__.py`, `verification/citation_integrity.py`, `verification/aggregation.py` | Deterministic quote checks and status rules |
| Modify | [graph_builder/graph_builder.py](../../graph_builder/graph_builder.py) | Build the new graph; keep the old graph available behind a flag until parity |
| New | `tests/test_aggregation.py`, `tests/test_citation_integrity.py`, `tests/test_routers.py`, `tests/test_research_graph.py`, `tests/test_domain_registry.py` | Unit and graph tests with fakes |
| Remove (end of phase) | [agents/supervisor_node.py](../../agents/supervisor_node.py), [agents/router.py](../../agents/router.py), [agents/llm_node.py](../../agents/llm_node.py), [agents/validation_node.py](../../agents/validation_node.py), [agents/validation_router.py](../../agents/validation_router.py) | After the new graph passes its tests and OQ-07 is confirmed |

**Acceptance criteria:**

1. For a seeded fixture, one document supporting claim A, one contradicting claim B and nothing for claim C, the graph yields A = `supported`, B = `contradicted`, C = `insufficient_evidence` → retry → `unresolved`.
2. Every claim status comes from `aggregation.py` rules. No status depends on model-reported confidence.
3. A judgment whose `supporting_quote` isn't found in the stored chunk is discarded, and the discard is recorded.
4. Targeted retry runs only for insufficient claims and never exceeds `max_retry_rounds`. Each budget limit (LLM calls, tool calls, time, cost) ends the run with status `partial` when exceeded.
5. The report narrative contains only supported claims. Every marker resolves to evidence with document, version and page (uploads) or URL and retrieval time (web).
6. A pharma question about efficacy marks those claims `needs_human_review`. Pharma `supported` requires two independent sources.
7. A procurement or insurance question is no longer rejected as out of scope. An off-topic question is rejected by the `general` scope rules.
8. All Phase 1 tests still pass.

---

## 5. Phase 3 — Persistence and ingestion abstraction

| Action | Path | Change |
|---|---|---|
| New | `persistence/__init__.py`, `persistence/db.py` | Engine and session factory from settings (`sqlite:///data/deepdive.db`) |
| New | `persistence/models.py` | The 13 MVP tables ([target-architecture.md §8.2](../architecture/target-architecture.md#82-minimum-viable-relational-model)) |
| New | `persistence/repositories.py` | Tenant-scoped, idempotent upserts with deterministic IDs |
| New | `alembic.ini`, `alembic/env.py`, `alembic/versions/0001_initial.py` | Initial migration; `render_as_batch=True` |
| Modify | [graph_builder/graph_builder.py](../../graph_builder/graph_builder.py) | Attach `SqliteSaver` (`data/checkpoints.sqlite`); `thread_id = run_id` |
| Modify | `agents/*_node.py` (new nodes) | Write through the repositories at node boundaries; write `agent_steps` |
| New | `ingestion/__init__.py`, `ingestion/connectors/__init__.py`, `ingestion/connectors/base.py` | `SourceConnector` protocol and models |
| New | `ingestion/connectors/local_upload.py` | `LocalUploadConnector`: hash-based IDs, versioning, uploader ACL |
| New | `ingestion/normalize.py` | `NormalizedDocument` |
| New | `ingestion/service.py` | `IngestionService`: extract → chunk → embed → index; job records; resume |
| New | `ingestion/object_store.py` | `LocalObjectStore` (`data/objects/`) behind an interface that a later GCS implementation can replace |
| Modify | [main_pipeline.py](../../main_pipeline.py) | Thin command-line wrapper over `IngestionService` |
| Modify | [pdf_processing/extractor.py](../../pdf_processing/extractor.py), [chunking.py](../../pdf_processing/chunking.py), [table_extractor.py](../../pdf_processing/table_extractor.py) | Add block offsets and section hints; deterministic chunk IDs; tables to Markdown chunks |
| New | `vector_database/milvus_index.py` | `VectorIndex` interface + `MilvusIndex` for `deepdive_chunks_v1`; index created once; upsert, deactivate, delete by version |
| Modify | [vector_database/milvus_setup.py](../../vector_database/milvus_setup.py) | Keep the connection helper; v1 collection left untouched |
| Modify | [retriever/retriever_pipeline.py](../../retriever/retriever_pipeline.py) | Permission-aware: tenant, ACL and `is_active` pre-filter + SQL post-check; one shared client |
| New | `tests/test_repositories.py`, `tests/test_local_upload_connector.py`, `tests/test_ingestion_service.py`, `tests/test_permission_filter.py`, `tests/test_migrations.py` | In-memory SQLite and fake vector index |
| New | `requirements.txt` additions | `sqlalchemy`, `alembic`, `langgraph-checkpoint-sqlite` (pinned) |

**Acceptance criteria:**

1. `alembic upgrade head` creates the schema on an empty SQLite file. `alembic downgrade base` reverses it.
2. Uploading the same PDF twice creates one document and one version; no vectors are duplicated. Uploading changed content under the same name creates version 2, and version 1's chunks are no longer returned.
3. A document uploaded by user A is never returned to user B (enforced by Milvus filter and SQL check; tested with fakes, and live against Zilliz).
4. A completed run can be reloaded from the database: plan, claims, verifications, evidence locators, report version, and per-node `agent_steps` with retry rounds.
5. Killing the process mid-run and resuming with the same `thread_id` completes without duplicate database rows.
6. An interrupted ingestion job resumes and skips versions already indexed.
7. Repository tests also pass against a PostgreSQL container (CI job, or a manual step until CI exists).

---

## 6. Phase 4 — Streamlit experience

| Action | Path | Change |
|---|---|---|
| Modify | [ui/streamlit_app.py](../../ui/streamlit_app.py) | Multipage entry point (`st.navigation`); cached graph and gateway |
| New | `ui/pages/research.py` | Question, domain, source scope, uploader, streamed progress, result tabs |
| New | `ui/pages/history.py` | Sessions and report versions; feedback |
| New | `ui/pages/sources.py` | Uploads tab (list, version, status, delete); connector tab placeholder behind a feature flag |
| New | `ui/components/progress.py`, `claims_table.py`, `evidence_panel.py`, `audit_panel.py` | Reusable widgets |
| New | `ui/source_scopes.py` | `SourceScope` registry (MVP: uploads, web) |
| Modify | [app.py](../../app.py) | Small wrapper that launches `ui/streamlit_app.py` (keeps `streamlit run app.py` working) |

**Acceptance criteria:**

1. A user can choose a domain and sources, ask a question, and watch node-level progress, including the retry round, while the run executes.
2. The Report, Claims, Evidence and Audit tabs show every claim's status. Each evidence item shows document title, page or URL, and the verified quote.
3. Disputed and unverified claims are visually distinct and never appear inside the main narrative.
4. Past sessions and report versions load from the database after an app restart.
5. Reports download as Markdown and JSON. The JSON validates against the report schema.
6. Feedback (rating and comment) is saved to `feedback` per report or claim.
7. Adding a fake `SourceScope` in a test makes it appear in the selector with no page changes.

---

## 7. Phase 5 — Evaluation, hardening, container

| Action | Path | Change |
|---|---|---|
| New | `evals/datasets/<domain>.jsonl`, `evals/fixtures/` | About 10 labelled cases per domain with seeded documents |
| New | `evals/run_eval.py` | Runs cases; writes `evals/results/<timestamp>.json` with metrics |
| Modify | [monitoring/logger.py](../../monitoring/logger.py) | JSON logs with `run_id` and `node`; stdout handler for containers; content-free messages |
| New | `monitoring/usage.py` | Usage and cost callback (if not already in the gateway) |
| New | `Dockerfile`, `.dockerignore` | Streamlit on 8080; non-root user; no secrets baked in |
| Move | [test_retriever.py](../../test_retriever.py) → `tests/integration/test_retriever_live.py` | Becomes a `@live` pytest test |
| Remove | `prompts/ prompt_template.py`, unused `Reranker.bm25_rerank` (or fix it) | Hygiene |
| New | `tests/security/test_prompt_injection.py` | A document containing "ignore previous instructions…" doesn't change routing, statuses or citations |

**Acceptance criteria:**

1. `python evals/run_eval.py` reports claim-status precision and recall, citation accuracy, unsupported-claim rate, retry yield, and mean cost and latency per domain.
2. Targets on the seeded set (to confirm, A-06): unsupported-claim rate in the narrative = 0; citation accuracy ≥ 95%; status precision ≥ 0.8.
3. `docker build` and `docker run` start the app with configuration from environment variables only, writing state only to mounted `data/`.
4. The prompt-injection test passes. Logs contain no document text.

---

## 8. Phase 6 — Basic multimodal (stretch)

Run this phase only if OQ-02 is resolved and a Mesh vision model is available.

| Action | Path | Change |
|---|---|---|
| Modify | [pdf_processing/image_extractor.py](../../pdf_processing/image_extractor.py) | Return stable `image_id` and bounding box |
| New | `pdf_processing/vision_captioner.py` | Mesh vision captions for charts and figures → `image_caption` chunks |
| New | `pdf_processing/scan_detector.py` | Flag pages with no text layer; optional render-to-vision transcription |
| Modify | `ingestion/service.py` | Optional multimodal stage; `CapabilityUnavailable` handled and recorded |

**Acceptance criteria:** a chart-based claim cites `image_id` and page. With the vision capability disabled, ingestion still succeeds, and the report notes that the image content wasn't analysed.

---

## 9. Phase 7+ — Post-MVP enterprise roadmap

| Track | Key work | Main files |
|---|---|---|
| Production persistence | Cloud SQL PostgreSQL, `PostgresSaver`, GCS object store, data migration script | `persistence/db.py`, `ingestion/object_store.py` (GCS implementation) |
| Deployment | Cloud Run service + Cloud Run Jobs/Tasks for ingestion; Secret Manager; CI/CD | `Dockerfile`, `deploy/` *(new)* |
| Connectors | `connector_configs` table; SharePoint/OneDrive (Graph delta), Google Drive, Confluence, S3/GCS/Azure Blob; scheduled incremental sync; webhooks | `ingestion/connectors/sharepoint.py` etc. |
| Identity and access | IAP or OIDC SSO; group resolution for ACLs; roles (admin, analyst, reviewer) | `persistence/models.py`, `ui/` |
| Multi-tenancy | `tenants` table; PostgreSQL row-level security; Milvus partition or collection per tenant; per-tenant config and quotas | `persistence/`, `vector_database/` |
| Governance | Hash-chained `audit_events`; `interrupt()` approval gates for high-risk reports; retention policies | `persistence/`, `graph_builder/` |
| Scale | Parallel `Send` fan-out for research; batch embeddings; ingestion queue | `agents/`, `ingestion/` |
| Advanced multimodal | Layout models, table question answering, OCR at scale | `pdf_processing/` |
| Operations | Dashboards, alerting, continuous evaluation, LangSmith datasets (if approved) | `monitoring/`, `evals/` |

---

## 10. Proposed dependency changes (not applied)

| Change | Package | Phase | Reason |
|---|---|---|---|
| Remove | `langchain-groq`, `langchain-anthropic`, `langchain_huggingface`, `sentence-transformers` | 1 | Direct inference outside Mesh |
| Remove | `faiss-cpu`, `chromadb`, `wikipedia`, `networkx`, `pyvis`, `matplotlib`, `plotly`, duplicate `pandas` | 1 | Unused (`plotly` may return for the UI if needed) |
| Keep | `langchain-openai` | 1 | Used **only** by `gateway/` as the OpenAI-compatible client for Mesh |
| Add | `pytest` | 1 | Tests |
| Add | `sqlalchemy`, `alembic`, `langgraph-checkpoint-sqlite` | 3 | Persistence |
| Add (post-MVP) | `psycopg`, `cloud-sql-python-connector`, `langgraph-checkpoint-postgres`, `google-cloud-storage`, `google-cloud-secret-manager` | 7 | Production on GCP |
| Review | `unstructured` | 3 | Heavy and currently unused; remove unless adopted for extraction |

All versions are to be pinned in Phase 1.

---

## 11. Schedule risks

| Risk | Mitigation |
|---|---|
| The chosen Mesh model ignores structured output | Probe at startup; fallback chain; schema validation with one repair retry; pick models with `supports_structured_output` |
| Verification multiplies LLM calls (claims × evidence) | Cap claims and evidence per claim; batch several evidence items per verifier call; budget routers |
| Zilliz cluster lacks array-field support | Fallback: store ACL principals as a delimited VARCHAR and filter with `like`, plus the mandatory SQL post-check |
| Hackathon rules disallow Tavily | Swap `retriever/web_search.py` to Mesh web search behind the same interface |
| Phase 2 is the largest phase | Build in this order: aggregation and integrity rules (pure functions, testable first) → nodes → wiring |
