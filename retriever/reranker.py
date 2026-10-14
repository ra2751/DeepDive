from langchain_community.retrievers import BM25Retriever
from langchain_core.vectorstores.utils import maximal_marginal_relevance

class Reranker:

    @staticmethod
    def bm25_rerank(documents, user_query):

        retriever = BM25Retriever.from_texts(
            [doc["text"] for doc in documents]
        )

        return retriever.get_relevant_documents(
            query=user_query,
            top_k=5
        )

    @staticmethod
    def mmr_rerank(query_embedding,
                   embeddings,
                   top_k=5):

        return maximal_marginal_relevance(
            query_embedding,
            embeddings,
            k=top_k
        )