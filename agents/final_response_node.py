from agents.state_node import AgentRagState
from monitoring.logger  import setup_logger
from langchain_core.messages import AIMessage

logger = setup_logger()

def finalResponseAgent(state:AgentRagState):
    """
    The FinalResponseAgent's role is to take the original query, the generated answer, and the validation results, 
    and produce a final response to the user. 
    If the validation indicates that the answer is not valid, this agent can modify the response accordingly (e.g., by adding disclaimers, suggesting further research, etc.).
    """

    question = state['query']
    final_answer = state['messages'][-1].content
    #validation_metadata = state['validation_metadata']

    #logger.info(f"FinalResponseAgent received question: {question}")
    logger.info(f"FinalResponseNode| Final Answer retrieved after Validation ")

    return {
        "messages": [AIMessage(content=final_answer)],
        "final_response": final_answer,
        "validation_metadata": {"is_valid": "yes", "reasoning": "Initial execution clean state."},
        "next_step": "",
        "loop_count": 0
    }