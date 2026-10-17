"""Scripted, offline 'world' for research-graph tests: Mesh fakes + read-only keyword Milvus fake."""

from tests.fakes import KeywordMilvusClient, ScriptedChat, TrackingEmbeddingsClient
from tests.research_fixtures import AUDIT, CONTRACT, PROCUREMENT_DOCS, SCORECARD, eid, judgment

QUESTION = "Review Acme Metals as a supplier: contract term, delivery performance and certifications."


def install_world(monkeypatch, settings):
    import agents.research_agent
    import agents.retry_agent
    import agents.verifier_agent
    import gateway.mesh
    from retriever.retriever_pipeline import RetrieverPipeline

    chats = {m: ScriptedChat(m) for m in (settings.mesh_model_supervisor, settings.mesh_model_generator,
                                         settings.mesh_model_verifier)}
    emb = TrackingEmbeddingsClient()
    monkeypatch.setattr(gateway.mesh, "_make_chat_client", lambda m, s, temperature=None: chats[m])
    monkeypatch.setattr(gateway.mesh, "_make_embeddings_client", lambda s: emb)

    class World:
        docs = list(PROCUREMENT_DOCS)
        planner = chats[settings.mesh_model_supervisor]
        generator = chats[settings.mesh_model_generator]
        verifier = chats[settings.mesh_model_verifier]
        embeddings = emb
        milvus = None

    def retriever(budget, s):
        World.milvus = World.milvus or KeywordMilvusClient(World.docs)
        return RetrieverPipeline(s.milvus_collection, budget=budget, settings=s, client=World.milvus)

    for mod in (agents.research_agent, agents.verifier_agent, agents.retry_agent):
        monkeypatch.setattr(mod, "make_retriever", retriever)
    return World


def plan(*subs, in_scope=True):
    return {"in_scope": in_scope, "scope_reason": "procurement question", "sub_questions": list(subs)}


def script_procurement(world):
    world.planner.structured_script = [plan("What is the Acme Metals contract term?",
                                            "How has Acme Metals performed and what certifications does it hold?")]
    world.generator.structured_script = [
        {"claims": [
            {"text": "The master supply agreement with Acme Metals runs until 31 December 2027.",
             "sub_question_index": 1, "claim_type": "contract_term", "evidence_ids": [eid(CONTRACT)]},
            {"text": "Acme Metals holds ISO 27001 certification.",
             "sub_question_index": 2, "claim_type": "compliance", "evidence_ids": [eid(AUDIT)]},
            {"text": "Acme Metals plans to open a new plant in Mexico in 2026.",
             "sub_question_index": 2, "claim_type": "fact", "evidence_ids": []},
            {"text": "Acme Metals delivered 98 percent of purchase orders on time during 2025.",
             "sub_question_index": 2, "claim_type": "supplier_performance", "evidence_ids": [eid(SCORECARD)]},
        ]},
        {"sentences": [
            {"text": "The Acme Metals master supply agreement runs until 31 December 2027.", "claim_ids": ["C1"]},
            {"text": "Acme Metals holds ISO 27001 certification.", "claim_ids": ["C2"]},
            {"text": "Acme Metals will remain the preferred supplier through 2030.", "claim_ids": ["C1"]},
        ]},
    ]
    world.verifier.structured_script = [
        {"judgments": [
            judgment("C1", CONTRACT, "supports", "runs until 31 December 2027"),
            judgment("C2", AUDIT, "contradicts", "Acme Metals does not hold ISO 27001 certification"),
            judgment("C4", SCORECARD, "supports", "Acme Metals delivered all purchase orders on time in 2025"),  # fabricated
        ]},
        {"judgments": []},
    ]
