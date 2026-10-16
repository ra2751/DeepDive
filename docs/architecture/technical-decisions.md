# DeepDive AI — Technical Decisions, Assumptions and Open Questions

| | |
|---|---|
| **Status** | Proposed — all decisions below are pending approval |
| **Date** | 2026-10-16 |
| **Related** | [Current assessment](architecture-assessment.md) · [Target architecture](target-architecture.md) · [Implementation roadmap](../roadmap/implementation-roadmap.md) |

This is the decision log for DeepDive. Each decision records its context, the choice, the alternatives considered and the consequences. Change a decision by adding a new entry that supersedes the old one; don't edit history.

**Statuses:** `Proposed` (awaiting approval) · `Accepted` · `Superseded by TD-xx`.

---

## 1. Technology choices

| Area | Choice | Basis | Notes |
|---|---|---|---|
| Language | Python | Mandated | — |
| Orchestration | LangGraph | Mandated; already used | See TD-01 |
| Model abstractions, retrieval, documents | LangChain (`langchain-core`, `langchain-openai` as the Mesh client, `langchain-text-splitters`) | Mandated; already used | `langchain-openai` used only inside `gateway/` |
| Inference gateway | Mesh API (OpenAI-compatible `/v1`) | Mandated | See TD-02 and §5 |
| Schemas / structured output | Pydantic v2 | Mandated; already used | — |
| UI | Streamlit (multipage) | Mandated; already used | See TD-14 |
| Operational DB (MVP) | SQLite | Mandated | See TD-09 |
| Operational DB (production) | PostgreSQL on GCP Cloud SQL | Mandated | — |
| ORM / migrations | SQLAlchemy 2.x, Alembic | Mandated | — |
| Checkpointing | `langgraph-checkpoint-sqlite` → `langgraph-checkpoint-postgres` | Recommended | Separate from the operational DB (TD-08) |
| Vector search | Milvus / Zilliz Cloud (existing) | Mandated "where practical" | See TD-10 |
| Web search | Tavily (existing) | Existing; pending OQ-01 | See TD-15 |
| PDF extraction | PyMuPDF (text, images), Camelot (tables) | Existing | `unstructured` currently unused |
| Testing | pytest | Mandated | — |
| Observability | Python logging (JSON), `agent_steps` table, optional LangSmith | Recommended | See TD-16 |
| Deployment target | Docker → GCP Cloud Run, Cloud SQL, GCS, Secret Manager | Mandated (future) | See TD-17 |
| Not adopted | CrewAI, AutoGen, other orchestrators; FAISS and Chroma (installed, unused) | — | See TD-01, TD-10 |

---

## 2. Decision log

### TD-01 — LangGraph is the only orchestration framework
- **Status:** Proposed
- **Context:** The workflow needs explicit typed state, conditional routing, bounded loops, checkpointing and future human-in-the-loop interrupts. LangGraph is mandated and already used.
- **Decision:** All agent orchestration uses LangGraph `StateGraph`. No CrewAI, AutoGen or custom orchestrators.
- **Alternatives:** CrewAI or AutoGen (role-based agents). Rejected: they add a second state model and weaker control over bounded loops, with no capability gap to justify them.
- **Consequences:** One mental model; checkpoints, `interrupt()` and `Send` are available for later phases.

### TD-02 — One Mesh gateway built on LangChain's OpenAI-compatible clients
- **Status:** Proposed
- **Context:** All runtime inference must go through Mesh API. Mesh documents an OpenAI-compatible `/v1` API with Bearer-key authentication. A sibling workspace project already uses `ChatOpenAI(base_url="https://api.meshapi.ai/v1")`. The current repository calls OpenAI and Gemini directly ([assessment §5](architecture-assessment.md#5-model-inference-integrations-observed)).
- **Decision:** Create `gateway/mesh.py` as the only module that constructs model clients, using `ChatOpenAI` and `OpenAIEmbeddings` pointed at Mesh. Models are chosen per role (planner, extractor, verifier, synthesizer, vision) from configuration. A compliance test forbids direct provider usage elsewhere.
- **Alternatives:**
  1. The native `meshapi` Python SDK. It isn't a LangChain `BaseChatModel`, so it would need a custom wrapper; kept as a fallback if OpenAI-compatibility gaps appear.
  2. Calling each model's own provider directly. Not allowed.
- **Consequences:** Swapping or adding models is a configuration change. Behaviour depends on the per-model capabilities Mesh exposes, which are probed at startup (§5). `OpenAIEmbeddings` must use `check_embedding_ctx_length=False` (to verify).

### TD-03 — No model-driven tool calling in the MVP
- **Status:** Proposed
- **Context:** Retrieved documents and web pages are untrusted input. Mesh tool-calling support varies by provider, and on Bedrock tools disable `response_format`.
- **Decision:** Graph nodes call retrieval tools deterministically. Models only produce structured outputs (plans, claims, judgments, queries, report text).
- **Alternatives:** A ReAct-style agent with tools. Deferred; it adds prompt-injection exposure and makes runs less predictable.
- **Consequences:** Behaviour is predictable and testable, and calls can be counted against the budget. Some research flexibility is lost, which parallel sub-questions and targeted retry partly make up for.

### TD-04 — Claim status is computed by rules from evidence, not by model confidence
- **Status:** Proposed
- **Context:** The current validator gives a single model's yes/no on the whole answer and pushes toward unsupported answers ([R-04](architecture-assessment.md#7-defects-and-risks-ranked)).
- **Decision:** Models output per-(claim, evidence) stances with quotes. `verification/aggregation.py` applies domain rules to produce `supported` / `contradicted` / `insufficient_evidence` / `unresolved`. Confidence scores are neither requested nor used.
- **Alternatives:** A model-assigned status, or confidence thresholds. Rejected: a model's confidence score is not verification.
- **Consequences:** Statuses can be explained (`rule_applied` is stored) and unit-tested. More model calls (mitigated by caps and batching).

### TD-05 — Independent verifier retrieval plus a deterministic quote check
- **Status:** Proposed
- **Context:** If the verifier sees only the extractor's chosen evidence, the check is circular, and models can invent citations.
- **Decision:** Each claim is searched again independently. Every judgment's `supporting_quote` must be found in the stored chunk text, or the judgment is discarded.
- **Consequences:** Invented citations are caught mechanically. Evidence must store text with stable chunk IDs and character offsets.

### TD-06 — Verifier model separate from generator model
- **Status:** Proposed
- **Context:** The existing code already used a different model (Gemini) for validation "to avoid bias".
- **Decision:** Keep that intent. The `verifier` role is set to a different model from `extractor`/`synthesizer`, both via Mesh.
- **Consequences:** Needs two Mesh models with structured-output support (OQ-03).

### TD-07 — Domains are configuration, not pipelines
- **Status:** Proposed
- **Context:** Five target domains; the current code hardcodes AI-only scope and blocks "finance".
- **Decision:** A YAML `DomainConfig` (scope, planning guidance, source preferences, web allow and block lists, claim types, verification policy, report template, disclaimer), validated by Pydantic, versioned, and recorded on each run.
- **Alternatives:** A subgraph per domain. Rejected: it duplicates the workflow.
- **Consequences:** A new domain needs no code. Very specialised domains can later add optional hook functions referenced from config.

### TD-08 — Four storage tiers; checkpoints are not the system of record
- **Status:** Proposed
- **Decision:** LangGraph checkpoints (resume and debug; can be purged), the operational DB (system of record), Milvus (rebuildable index) and the object store (raw files) are separate. Checkpoints use a separate SQLite file in the MVP and a separate schema in PostgreSQL.
- **Alternatives:**
  1. Read history from checkpoints. Rejected: the format is internal and unsuitable for querying or audit.
  2. Share one SQLite file. Rejected: it mixes migration ownership between Alembic and the checkpointer library.
- **Consequences:** Repository writes must be idempotent (TD-13).

### TD-09 — SQLAlchemy 2 + Alembic from the first persistence commit
- **Status:** Proposed
- **Decision:** Portable types only; Alembic with `render_as_batch=True`; database URL from settings; repository tests against SQLite and PostgreSQL.
- **Consequences:** Moving from SQLite to Cloud SQL is a configuration change plus an optional data copy.

### TD-10 — Keep Milvus/Zilliz with a new, versioned collection
- **Status:** Proposed
- **Context:** The existing `ai_research_assistant` collection has no IDs, versions, tenant or ACL fields, and its vectors came from calling OpenAI directly.
- **Decision:** Create `deepdive_chunks_v1` (schema in [target §8.3](target-architecture.md#83-milvus-v2-collection)), re-embed via Mesh (`openai/text-embedding-3-small`, 1536 dimensions), and leave the old collection untouched.
- **Alternatives:**
  1. Mesh "Files & RAG". Less control over ACL filtering and locators; vendor coupling.
  2. pgvector on Cloud SQL. Fewer systems to run, but a migration away from existing code.
  3. Vertex AI Vector Search. More operational overhead for the MVP.
  4. FAISS or Chroma (installed). Not durable or shared.
- **Consequences:** Needs Milvus 2.4+ features (OQ-04). The embedding model is recorded per chunk so re-embedding is safe.

### TD-11 — Pluggable connector interface; only the upload connector in the MVP
- **Status:** Proposed
- **Decision:** `SourceConnector` protocol (discover, full_listing, changes_since, fetch, get_acl) plus normalization, extraction, chunking, embedding and indexing stages. The MVP implements `LocalUploadConnector` only.
- **Consequences:** SharePoint, Drive and other connectors are new classes plus a `connector_configs` table, with no change to the workflow.

### TD-12 — Permissions are enforced twice and inherited from the source
- **Status:** Proposed
- **Decision:** ACL principals from the source are stored on each chunk. Retrieval pre-filters by tenant, active flag and the user's principals in Milvus, then re-checks against the SQL ACL. Upload ACL = the uploader only.
- **Alternatives:** Late binding only (check against the source at query time). Better accuracy but slow and rate-limited; reserved for high-sensitivity sources later.
- **Consequences:** ACL staleness is bounded by sync frequency; documented as a residual risk for connectors.

### TD-13 — Deterministic IDs and idempotent writes
- **Status:** Proposed
- **Decision:**
  - `document_id = sha256(tenant, source_system, external_id)`
  - `version_id = sha256(content)`
  - `chunk_id = sha256(version_id, element_index, chunk_index)`
  - evidence, claim and verification IDs are derived from the run and their content
  - all writes are upserts
- **Consequences:** Safe replay after resuming from a checkpoint, deduplicated uploads, and stable provenance links.

### TD-14 — Streamlit multipage UI with a SourceScope registry
- **Status:** Proposed
- **Decision:** Research, History and Knowledge Sources pages. The Sources selector is fed by a registry so connectors appear without UI redesign. Progress is streamed with `graph.stream(stream_mode="updates")`.
- **Consequences:** `app.py` becomes a small wrapper.

### TD-15 — Keep Tavily for web search behind an interface (pending rules)
- **Status:** Proposed — depends on OQ-01
- **Decision:** Move Tavily into `retriever/web_search.py` behind a `WebSearch` interface with per-domain allow and block lists and `retrieved_at`.
- **Alternatives:** Mesh web search endpoint (listed in the Mesh SDK). Adopt if the rules require all external AI services to go through Mesh.

### TD-16 — LangSmith optional and off by default
- **Status:** Proposed
- **Decision:** Core observability is JSON logs plus `agent_steps`. LangSmith is enabled only by configuration, with input and output masking where documents are confidential.
- **Consequences:** No third-party trace export of enterprise content unless explicitly approved.

### TD-17 — Containers first; Cloud Run + Cloud SQL + GCS + Secret Manager as the production target
- **Status:** Proposed
- **Decision:** One Dockerfile with configuration from the environment only. Production on Cloud Run with session affinity, Cloud SQL, GCS, Secret Manager, and Cloud Run Jobs or Cloud Tasks for ingestion.
- **Consequences:** SQLite and local files are for development only. The same container ports to AWS (App Runner/ECS) or Azure (Container Apps).

### TD-18 — Minimal MVP schema; defer SaaS-only tables
- **Status:** Proposed
- **Decision:** 13 MVP tables. The plan is stored as `plan_json` on `research_runs`. Deferred: `tenants`, `connector_configs`, `evaluation_results`, `audit_events`, `document_chunks`, `research_plans`. `tenant_id` columns exist from day one.
- **Consequences:** Adding tenancy later is additive (new table + foreign key + row-level security) rather than a reshaping.

### TD-19 — Fix the existing loop bound before replacing the graph
- **Status:** Proposed
- **Context:** [R-02 and R-03](architecture-assessment.md#7-defects-and-risks-ranked): the current graph loops until LangGraph's recursion limit stops it, and has a typo'd branch name.
- **Decision:** Phase 1 fixes the bound and the branch so the existing app is safe to demo while Phase 2 is built.

---

## 3. Assumptions

| ID | Assumption | If wrong |
|---|---|---|
| A-01 | Mesh's OpenAI-compatible endpoint works with `langchain-openai` `ChatOpenAI` and `OpenAIEmbeddings` (seen in a sibling project, not yet tested here) | Wrap the native `meshapi` SDK as a custom LangChain model (TD-02 alternative 1) |
| A-02 | At least two Mesh chat models with reliable `json_schema` structured output are available to our key | Use function-calling or JSON-mode fallback with stricter validation |
| A-03 | `openai/text-embedding-3-small` via Mesh returns 1536-dimension vectors (as documented) | Adjust the Milvus v2 dimension before the collection is created |
| A-04 | Tavily is permitted as a non-inference tool | Swap to Mesh web search (TD-15) |
| A-05 | A single local user with `tenant_id="default"` is acceptable for the MVP | Add basic auth (e.g. a Streamlit auth proxy) in Phase 4 |
| A-06 | Default budgets: 5 sub-questions, 2 retry rounds, 25 claims, 40 LLM calls, 30 tool calls, 60 s per call, 300 s and US$0.50 per run; eval targets as in roadmap Phase 5 | Tune in domain config and settings |
| A-07 | Pharma and insurance high-risk claims need ≥2 independent sources; other domains need 1 | Change `verification.min_independent_sources` per domain |
| A-08 | Answers from model knowledge alone are out of scope as evidence; that knowledge may guide planning but never supports a claim | Re-introduce it as a clearly labelled, unverified "background" section |
| A-09 | The existing Zilliz cluster can host a second collection, and re-embedding the current PDFs is acceptable | Reuse the existing collection temporarily, without ACL support |
| A-10 | Existing hackathon data in `data/` and Milvus does not need migrating | Write a one-off backfill script |

---

## 4. Open questions

| ID | Question | Blocks | Owner |
|---|---|---|---|
| OQ-01 | Do the hackathon rules allow Tavily (a non-model search API), or must web search go through Mesh? | Phase 1 (A-04), TD-15 | Product / hackathon rules |
| OQ-02 | Is non-model OCR (e.g. Tesseract) allowed, or must OCR and vision go through a Mesh vision model? | Phase 6 | Product / hackathon rules |
| OQ-03 | Which Mesh models does our key allow for chat, verifier and embeddings? Do they report `supports_structured_output` and image input? Does Mesh return cost per request? | Phases 1–2; cost budget enforcement | Engineering (live probe) |
| OQ-04 | What version and tier is the Zilliz cluster? Does it support array fields, `ARRAY_CONTAINS_ANY` and partition keys? | Phase 3 | Engineering |
| OQ-05 | What are Mesh's data retention, logging, training-use and regional processing terms? No documentation page was found. | Use with confidential enterprise documents | Legal / vendor management |
| OQ-06 | Confirm the per-domain verification policy (min sources, high-risk claim types, disclaimers) for pharma and insurance | Phase 2 domain configs | Domain leads |
| OQ-07 | May the old supervisor, model-knowledge-only, validator and finalize nodes be removed once the new graph passes its tests? | End of Phase 2 | Project owner |
| OQ-08 | Is LangSmith approved, and under which data-handling terms? | Optional tracing | Security |
| OQ-09 | Which identity provider will enterprise customers use (Entra ID, Google Workspace, Okta)? | Post-MVP identity and connectors | Product |
| OQ-10 | What is the hackathon deadline? It sets which stretch phase (6) is attempted. | Planning | Project owner |

---

## 5. Mesh API capability verification

Sources were reviewed on 2026-10-16:

- the Mesh developer docs (`developers.meshapi.ai`), including the `llms.txt` index and the structured-output, embeddings and images/vision pages;
- the `meshapi` PyPI page.

Items marked "to verify" must be confirmed with a live key in Phase 1.

| Capability | Documented behaviour | Design impact | Verification |
|---|---|---|---|
| Authentication | `Authorization: Bearer rsk_…` (Router Service Key) | `MESH_API_KEY` in settings and Secret Manager | Live call |
| Protocol | OpenAI-compatible, base URL `https://api.meshapi.ai/v1`; model IDs look like `provider/model` (e.g. `openai/gpt-4o-mini`) | `ChatOpenAI(base_url=…)` | Live call through LangChain (A-01) |
| Chat completions | `POST /v1/chat/completions` | Primary API | Live |
| Responses API | Available (reasoning effort, provider-hosted tools) | Not used in the MVP | — |
| Streaming | SSE; the SDK does not retry streams | UI progress uses LangGraph events, not token streaming; token streaming optional | Live, including usage while streaming |
| Structured output | `response_format` `json_schema` (name + schema) and `json_object`; support varies by model; DeepSeek, Arcee and ILMU only `json_object`; some models **silently ignore** it and return text with `finish_reason: "stop"`; on Bedrock it is dropped when `tools` are present; per-model `supports_structured_output` flag in `GET /v1/models` | Always validate with Pydantic; fallback chain; startup probe; no tools in the same call (TD-03) | Per configured model |
| Tool calling | Documented capability page (not reviewed in detail) | Not needed in the MVP | Defer |
| Embeddings | `POST /v1/embeddings`; examples `openai/text-embedding-3-small` (1536), `-3-large` (3072), `amazon/titan-embed-text-v2` (1024); optional `dimensions`; batch limits not stated | Matches the existing 1536-dimension schema; batch size configurable | Live, plus batch size limits |
| Vision | `image_url` content parts on chat completions, base64 data URLs supported; per-model `input_modalities` | Phase 6 captions and OCR fallback | Per model |
| OCR / PDF input | Not documented | OCR = render the page to an image + vision model, or a non-model tool (OQ-02) | — |
| Files & RAG, web search | Mesh offers file upload, RAG and web search endpoints | Not adopted (TD-10, TD-15); a possible alternative | — |
| Rate limits / spend caps | Per key, team and organisation; 429 on RPM/RPD; spend caps (402); live-usage endpoint; numbers not stated | Budget routers; map 402 to `partial` | Read the limits endpoint |
| Retry / fallback | The SDK retries 429/502/503/504 with exponential backoff; the platform can retry or reroute upstream | Record the **model served** per call for audit | Check the response `model` field |
| Errors | 401, 402, 403, 404 (`model_not_found`), 422, 429, 500, `stream_interrupted`; `request_id` returned | Typed error mapping in the gateway; log `request_id` | Live |
| Data retention / privacy | **No page found** | Blocker for confidential enterprise data (OQ-05) | Vendor confirmation |
| LangChain integration | No dedicated page | Relies on OpenAI compatibility (A-01) | Phase 1 live tests |

If Mesh lacks a capability we need, the gap is documented and the workflow degrades in a way it records. **Requests are never routed to another provider outside Mesh.**
