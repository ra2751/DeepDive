from gateway.mesh import embeddings as mesh_embeddings


class EmbeddingGenerator:
    """Embeddings via Mesh API only.

    Without an explicit budget, an ingestion budget is used (separate from
    research-run budgets).
    """

    def __init__(self, budget=None, settings=None):
        from budget import new_ingestion_budget
        from config.settings import get_settings

        settings = settings or get_settings()
        self.budget = budget or new_ingestion_budget(settings)
        self.embedding_model = mesh_embeddings(self.budget, settings)

    def generate_embeddings(self, texts):
        return self.embedding_model.embed_documents(texts)
