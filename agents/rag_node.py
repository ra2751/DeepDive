from retriever.retriever_pipeline import RetrieverPipeline
from agents.state_node import AgentRagState
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.messages import AIMessage
from config.config import Config
from agents.state_node import AgentRagState
from monitoring.logger import setup_logger

logger = setup_logger()

def format_docs(docs):
    return "\n\n".join(doc['text'] for doc in docs)

def RagAgent(state:AgentRagState):

    """
    The RagAgent's role is to take the routed query and perform retrieval-augmented generation.
    It retrieves relevant documents based on the query and then generates a response 
    using those documents as context.
    """

    # Extract the query and retrieved documents from the state
    question = state['query']

    logger.info(f"RagAgent received query: {question}")

    retriever = RetrieverPipeline("ai_research_assistant")
    docs = retriever.search(question)

    logger.info(f"RagAgent received retrieved documents: {docs}")

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

    # Get the language model from config
    config = Config()
    model = config.get_llm()

    # Combine the prompt and model into a single chain
    chain = prompt | model | StrOutputParser()  # Use a simple string output parser for the final response

    response = chain.invoke({"question": question, "formatted_docs": formatted_docs})

    logger.info(f"RagAgent generated response successfully")

    retrieved_docs = [{"content": doc['text'], "source": doc['source']} for doc in docs]

    logger.info(
            f"RagAgent | Retrieved Docs Count: {len(retrieved_docs)} | Response Length: {len(response)}"
            )
    
    return {
        "retrieved_docs": retrieved_docs,
        "messages": [AIMessage(content=response)]
        
    }
