"""T05: importing modules needs no keys and makes no network calls."""

import importlib

import pytest

MODULES = [
    "config.config", "config.settings", "gateway", "gateway.mesh", "budget", "budget.manager",
    "retriever.web_search", "retriever.retriever_pipeline", "retriever.reranker",
    "pdf_processing.embeddings", "pdf_processing.extractor", "pdf_processing.chunking",
    "vector_database.indexing", "vector_database.milvus_setup", "main_pipeline",
    "agents.state_node", "agents.supervisor_node", "agents.rag_node", "agents.web_node",
    "agents.llm_node", "agents.validation_node", "agents.validation_router", "agents.router",
    "agents.final_response_node", "graph_builder.graph_builder", "api.main",
]


@pytest.mark.parametrize("module", MODULES)
def test_module_imports_offline(module):
    importlib.import_module(module)


def test_build_graph_compiles_offline():
    from graph_builder.graph_builder import build_graph

    graph = build_graph()
    assert {"supervisor", "rag_agent", "web_search_agent", "general_llm_agent", "validator", "finalize"} <= set(graph.nodes)
