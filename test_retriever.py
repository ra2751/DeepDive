
import logging
from retriever.retriever_pipeline import RetrieverPipeline

def run_retriever_test():
    # Initialize the retriever pipeline with the name of your Milvus collection

    print(f"--- Initializing Retriever for ai_research_assistant ---")
    try:
        retriever = RetrieverPipeline(collectionName="ai_research_assistant")
    except Exception as e:
        print(f"Error initializing RetrieverPipeline: {e}")
        return
    
    # Define a sample query
    test_query = "What is the purpose of the attention mechanism in transformers?"
    
    print(f"\n--- Running Search: '{test_query}' ---")

    try:
        # Perform the search using the retriever pipeline
        results = retriever.search(query=test_query, top_k=5)

        if not results:
            print("No results found.")
            return
        else:
            print(f"Retrieved {len(results)} relevant chunks.\n")
        
        for idx, doc in enumerate(results):
            print(f"Result {idx+1}:")
            print(f"Text: {doc['text']}")
            print(f"Source: {doc['source']}")
            print(f"Page: {doc['page']}")
            print("-" * 50)
    
    except Exception as e:
        print(f"Error during search: {e}")
        return
    
if __name__ == "__main__":
    run_retriever_test()  
