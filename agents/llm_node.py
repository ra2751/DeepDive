from monitoring.logger import setup_logger
from langchain_core.messages import AIMessage
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableConfig
from agents.state_node import AgentRagState
from budget import BudgetExceeded, budget_from_config
from gateway import MeshError, chat_model

logger = setup_logger()

def GeneralLLMAgent(state:AgentRagState, config: RunnableConfig):
    """
    Role: General purpose reasoning machine. Handles logic, coding, and brainstorming
    tasks that do not require external document lookups.

    Answers from this agent have no sources and are always labelled "ungrounded".
    """
    question = state['query']
    budget = budget_from_config(config)
    logger.info("GeneralLLMAgent| Query received")

    template = """
    You are a helpful, logical, and highly accurate AI assistant. 
    Answer the user's query directly using your built-in knowledge.

    User Query: {question}
    """
    
    prompt = PromptTemplate.from_template(template)
    model = chat_model("generator", budget)
    chain = prompt | model | StrOutputParser()

    try:
        response = chain.invoke({"question": question})
    except BudgetExceeded as e:
        return {"termination_reason": e.reason, "retrieved_docs": [], "draft_answer": None}
    except MeshError as e:
        logger.error(f"GeneralLLMAgent | Mesh call failed: {type(e).__name__}")
        return {"termination_reason": "mesh_error", "retrieved_docs": [], "draft_answer": None,
                "notices": [f"Answer generation failed: {e}"]}
    logger.info(f"GeneralLLMAgent| response generated successfully")

    return {
        "retrieved_docs": [],  # Standardized key: Empty because no documents were used
        "messages": [AIMessage(content=response)],
        "draft_answer": response,
    }
