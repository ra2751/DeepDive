from retriever.retriever_pipeline import RetrieverPipeline
from agents.state_node import AgentRagState
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.messages import AIMessage
from langchain_core.runnables import RunnableConfig
from budget import BudgetExceeded, budget_from_config
from config.settings import get_settings
from gateway import MeshError, chat_model
from monitoring.logger import setup_logger

logger = setup_logger()

def format_docs(docs):
    return "\n\n".join(doc['text'] for doc in docs)

def RagAgent(state:AgentRagState, config: RunnableConfig):

    """
    The RagAgent's role is to take the routed query and perform retrieval-augmented generation.
    It retrieves relevant documents based on the query and then generates a response 
    using those documents as context.
    """

    # Extract the query and retrieved documents from the state
    question = state['query']
    budget = budget_from_config(config)
    settings = get_settings()
    notices = []

    logger.info("RagAgent received query")

    try:
        retriever = RetrieverPipeline(settings.milvus_collection, budget=budget, settings=settings)
        docs = retriever.search(question)
    except BudgetExceeded as e:
        return {"termination_reason": e.reason, "retrieved_docs": [], "draft_answer": None}
    except Exception as e:  # Milvus unavailable, incompatible embeddings, Mesh failure, ...
        logger.error(f"RagAgent | Document retrieval failed: {type(e).__name__}")
        docs = []
        notices.append(f"Document search unavailable: {type(e).__name__}: {e}")

    if settings.milvus_embedding_compat == "unverified" and docs:
        notices.append(
            "Document search used Mesh query embeddings against an index whose embedding "
            "compatibility has not yet been verified (pending the G2 compatibility check)."
        )

    logger.info(f"RagAgent retrieved {len(docs)} document chunks")

    # Format the retrieved documents into a string for the prompt
    formatted_docs = format_docs(docs)

    # Create a prompt template that incorporates the query and retrieved documents
    template = """
    You are an AI assistant that provides information based on retrieved documents. 
    Use the following retrieved documents to answer the user's query.
    If you don't know the answer, just say that you don't know.

    \nUser Query: {question}

    \nRetrieved Documents:
    {formatted_docs}

    Please provide a comprehensive answer based on the above information.
    """

    prompt = PromptTemplate.from_template(template)

    # Language model via the Mesh API gateway
    model = chat_model("generator", budget)

    # Combine the prompt and model into a single chain
    chain = prompt | model | StrOutputParser()  # Use a simple string output parser for the final response

    retrieved_docs = [{"content": doc['text'], "source": doc['source'], "page": doc.get('page')} for doc in docs]

    try:
        response = chain.invoke({"question": question, "formatted_docs": formatted_docs})
    except BudgetExceeded as e:
        return {"termination_reason": e.reason, "retrieved_docs": retrieved_docs, "draft_answer": None, "notices": notices}
    except MeshError as e:
        logger.error(f"RagAgent | Mesh call failed: {type(e).__name__}")
        return {"termination_reason": "mesh_error", "retrieved_docs": retrieved_docs, "draft_answer": None,
                "notices": notices + [f"Answer generation failed: {e}"]}

    logger.info(
            f"RagAgent | Retrieved Docs Count: {len(retrieved_docs)} | Response Length: {len(response)}"
            )
    
    return {
        "retrieved_docs": retrieved_docs,
        "messages": [AIMessage(content=response)],
        "draft_answer": response,
        "notices": notices,
    }
