from langchain_openai.embeddings import OpenAIEmbeddings

class EmbeddingGenerator:

    def __init__(self):
        self.embedding_model = OpenAIEmbeddings(
            model="text-embedding-3-small"
        )

    def generate_embeddings(self, texts):
        return self.embedding_model.embed_documents(texts)