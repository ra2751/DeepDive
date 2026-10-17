"""T21: document text never appears in logs."""

import logging

from tests.graph_helpers import budget, patch_retriever, route, run_graph, verdict

SECRET = "CONFIDENTIAL-FIXTURE-7731 supplier pricing schedule"


def test_document_text_not_logged(fake_mesh, monkeypatch, caplog):
    patch_retriever(monkeypatch, docs=[{"text": SECRET, "source": "contract.pdf", "page": 1}])
    fake_mesh.supervisor.script = [route("rag_agent")]
    fake_mesh.verifier.structured_script = [verdict("yes")]
    with caplog.at_level(logging.INFO, logger="ai_research_assistant"):
        run_graph(budget())
    assert "CONFIDENTIAL-FIXTURE-7731" not in caplog.text
    assert "MeshGateway" in caplog.text  # model calls are logged (requested/served/tokens)
