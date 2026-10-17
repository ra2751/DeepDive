import json

from budget import BudgetLimits, BudgetManager
from graph_builder.graph_builder import build_graph
from retriever.retriever_pipeline import RetrieverPipeline
from tests.fakes import FakeMilvusClient

DOCS = [
    {"text": "Multi-head attention runs several attention heads in parallel.", "source": "attention.pdf", "page": 3},
    {"text": "Each head projects queries, keys and values separately.", "source": "attention.pdf", "page": 4},
]


def route(next_route, topic="AI Transformer Related"):
    return json.dumps({"topic": topic, "reasoning": "test routing", "next_route": next_route})


def verdict(is_valid):
    return {"is_valid": is_valid, "reasoning": f"test verdict {is_valid}"}


def patch_retriever(monkeypatch, docs=DOCS):
    import agents.rag_node

    milvus = FakeMilvusClient(docs)

    def factory(name, budget=None, settings=None):
        return RetrieverPipeline(name, budget=budget, settings=settings, client=milvus)

    monkeypatch.setattr(agents.rag_node, "RetrieverPipeline", factory)
    return milvus


def run_graph(budget, query="Explain multi-head attention in the attached papers"):
    state = {
        "query": query,
        "messages": [],
        "loop_count": 0,
        "validation_metadata": {"is_valid": "pending", "reasoning": "", "performed": False},
        "draft_answer": None,
        "out_of_scope": False,
        "termination_reason": None,
        "notices": [],
    }
    return build_graph().invoke(state, config={"configurable": {"budget": budget}, "recursion_limit": 40})


def budget(**overrides):
    return BudgetManager(BudgetLimits(**overrides))
