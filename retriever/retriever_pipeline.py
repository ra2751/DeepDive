import time

from monitoring.logger import setup_logger
from pymilvus import Collection, connections, MilvusClient
from pdf_processing.embeddings import EmbeddingGenerator
from retriever.reranker import Reranker
import os
from dotenv import load_dotenv
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

"""

load_dotenv()

logger = setup_logger()

class RetrieverPipeline:
    
    # This class is responsible for retrieving relevant information 
    # from a Milvus vector database based on a given query.
    # The class uses an embedding model to convert the query 
    # into a vector representation and then performs a search

    def __init__(self, collectionName:str):
        self.embedding_generator = EmbeddingGenerator()
        self.embedding_model = self.embedding_generator.embedding_model
        self.uri = os.getenv("Milvus_ENDPOINT")
        self.token = os.getenv("Milvus_API_KEY")
        self.collection_name = collectionName
        try:
            self.client = MilvusClient(uri=self.uri, token=self.token)
            logger.info("Connected to Milvus Cloud successfully")
            self.client.load_collection(collection_name=self.collection_name)
            logger.info(f"Collection '{self.collection_name}' loaded successfully")
        except Exception as e:
            logger.error(f"Failed to connect to Milvus Cloud: {e}")
            raise e
        

    def search(self, query:str, top_k:int=5):
        start_time = time.time()

        raw_embedding = self.embedding_model.embed_query(query)
        query_embedding = list(raw_embedding) # Convert to list for Milvus search

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

        candidate_docs = []
        candidate_embeddings = []

        for hit in results[0]:
            candidate_docs.append({
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
            top_k=top_k
        )

        final_results = [candidate_docs[i] for i in selected_indices]

        end_time = time.time()

        retrieval_time = end_time - start_time
        
        logger.info(
            f"Retriever | MMR applied | Selected {top_k} from {fetch_k} | Time: {retrieval_time:.4f}s"
            )

        return final_results