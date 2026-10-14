from monitoring.logger import setup_logger
from langchain_core.messages import AIMessage
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from agents.state_node import AgentRagState
from config.config import Config

logger = setup_logger()

def GeneralLLMAgent(state:AgentRagState):
    """
    Role: General purpose reasoning machine. Handles logic, coding, and brainstorming
    tasks that do not require external document lookups.
    """
    question = state['query']
    logger.info(f"GeneralLLMAgent| Query: {question}")

    template = """
    You are a helpful, logical, and highly accurate AI assistant. 
    Answer the user's query directly using your built-in knowledge.

    User Query: {question}
    """
    
    prompt = PromptTemplate.from_template(template)
    config = Config()
    model = config.get_llm()
    chain = prompt | model | StrOutputParser()

    response = chain.invoke({"question": question})
    logger.info(f"GeneralLLMAgent| response generated successfully")

    return {
        "retrieved_docs": [],  # Standardized key: Empty because no documents were used
        "messages": [AIMessage(content=response)]
    }
