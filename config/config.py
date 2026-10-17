"""
Backward-compatible configuration facade.

All model access now goes through the Mesh API gateway (gateway/mesh.py); this
class no longer constructs provider clients or writes API keys into os.environ.
Use config.settings.get_settings() for typed settings.
"""

from config.settings import get_settings


class Config:
    COLLECTION_NAME = "ai_research_assistant"
    TOP_K = 5

    # Document Processing
    CHUNK_SIZE = 500
    CHUNK_OVERLAP = 50 

    @classmethod
    def get_llm(cls, budget):
        """Generator chat model via Mesh, metered by the given BudgetManager."""
        from gateway import chat_model
        return chat_model("generator", budget, get_settings())

    @classmethod
    def get_llm_validation(cls, budget):
        """Verifier chat model via Mesh, metered by the given BudgetManager."""
        from gateway import chat_model
        return chat_model("verifier", budget, get_settings())
