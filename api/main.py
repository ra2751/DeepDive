"""
Minimal FastAPI service. It exposes a liveness endpoint; research runs through the
Streamlit UI (app.py) and the shared graph in graph_builder/research_graph.py.

Run locally:  uvicorn api.main:app --host 127.0.0.1 --port 8000
"""

from fastapi import FastAPI

from api import __version__

app = FastAPI(title="DeepDive AI API", version=__version__)


@app.get("/health")
def health() -> dict:
    """Liveness only: no external calls, no authentication required."""
    return {"status": "ok", "service": "deepdive-api", "version": __version__}
