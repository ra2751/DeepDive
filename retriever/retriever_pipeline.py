import time

from monitoring.logger import setup_logger
from pymilvus import MilvusClient
from retriever.reranker import Reranker
import numpy as np


"""
This module defines the RetrieverPipeline class, 
which is responsible for retrieving relevant information 
from a Milvus vector database based on a given query. 
The class uses an embedding model to convert 
the query into a vector representation and 
then performs a search in the Milvus collection 
to find the most relevant documents. The retrieved documents 
are then reranked using Maximal Marginal Relevance (MMR) 
to ensure diversity in the results. 

Query embeddings are generated through the Mesh API gateway and every
search is counted against the run's BudgetManager.
"""

logger = setup_logger()


class EmbeddingCompatibilityError(RuntimeError):
    """Mesh embeddings were found incompatible with the vectors stored in the collection."""


class RetrieverPipeline:
    
    # This class is responsible for retrieving relevant information 
    # from a Milvus vector database based on a given query.
    # The class uses an embedding model to convert the query 
    # into a vector representation and then performs a search

    def __init__(self, collectionName: str, budget=None, embeddings=None, client=None, settings=None):
        from budget import new_research_budget
        from config.settings import get_settings

        self.settings = settings or get_settings()
        self.collection_name = collectionName
        self.budget = budget or new_research_budget(self.settings)

        if self.settings.milvus_embedding_compat == "incompatible":
            # Never silently query existing vectors with an incompatible embedding model.
            raise EmbeddingCompatibilityError(
                f"Mesh embeddings are marked incompatible with collection '{collectionName}'. "
                "Re-embed into a new collection before using document search."
            )

        if embeddings is None:
            from pdf_processing.embeddings import EmbeddingGenerator
            embeddings = EmbeddingGenerator(budget=self.budget).embedding_model
        self.embedding_model = embeddings

        if client is None:
            self.settings.require("milvus_endpoint", "milvus_api_key")
            try:
                client = MilvusClient(uri=self.settings.milvus_endpoint, token=self.settings.milvus_api_key)
                logger.info("Connected to Milvus Cloud successfully")
                client.load_collection(collection_name=self.collection_name)
                logger.info(f"Collection '{self.collection_name}' loaded successfully")
            except Exception as e:
                logger.error(f"Failed to connect to Milvus Cloud: {type(e).__name__}")
                raise
        self.client = client
        

    def search(self, query:str, top_k:int=5):
        """Embed one query (1 Mesh embedding request) and search Milvus."""
        self.budget.check("tool")
        raw_embedding = self.embedding_model.embed_query(query)
        return self._search_vector(list(raw_embedding), top_k)

    def search_many(self, queries: list, top_k: int = 3) -> list:
        """Embed several queries in ONE batched Mesh request, then search Milvus per query.

        Returns a list (one per query) of result lists.
        """
        if not queries:
            return []
        for _ in queries:
            self.budget.check("tool")
        vectors = self.embedding_model.embed_documents(list(queries))
        return [self._search_vector(list(v), top_k) for v in vectors]

    def _search_vector(self, query_embedding: list, top_k: int):
        start_time = time.time()

        # Count the vector search as a tool call before any work is done
        self.budget.check("tool")
        self.budget.record_tool("milvus_search")

        # For Milvus, the search parameters depend on the index type used.
        search_params = {
            "metric_type": "L2",
            "params": {
                "ef": 64
            }
        }

        fetch_k = 10 # Number of candidates to fetch for MMR reranking
        
        results = self.client.search(
            collection_name=self.collection_name, # Name of the collection to search
            data=[query_embedding], # List of query embeddings
            anns_field="embedding", # Which field contains the embeddings
            search_params=search_params, 
            limit=fetch_k, # Number of top results to retrieve
            output_fields=["text","source","page","embedding"] # What data to bring back (e.g., original text chunks)
        )

        hits = results[0] if results else []
        if not hits:
            logger.info("Retriever | No candidates returned from Milvus")
            return []

        candidate_docs = []
        candidate_embeddings = []

        for hit in hits:
            candidate_docs.append({
                "id": hit.get("id"), # Milvus primary key (chunk provenance)
                "score": hit.get("distance"), # Similarity score (distance) from the query
                "text": hit.entity.get("text"), # Original text chunk from the document
                "source": hit.entity.get("source"), # Source document identifier (e.g., filename or URL)
                "page": hit.entity.get("page") # Page number in the source document, if applicable
            })    
            # The embedding is needed for MMR reranking, so we also store it in a separate list
            candidate_embeddings.append(hit.entity.get("embedding"))

        # Convert to numpy arrays for MMR reranking
        query_vec = np.array(query_embedding) 
        candidate_vecs = np.array(candidate_embeddings)

        selected_indices = Reranker.mmr_rerank(
            query_embedding=query_vec,
            embeddings=candidate_vecs,
            top_k=min(top_k, len(candidate_docs))
        )

        final_results = [candidate_docs[i] for i in selected_indices]

        end_time = time.time()

        retrieval_time = end_time - start_time
        
        logger.info(
            f"Retriever | MMR applied | Selected {len(final_results)} from {len(candidate_docs)} | Time: {retrieval_time:.4f}s"
            )

        return final_results
