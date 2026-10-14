from monitoring.logger import setup_logger

# import the shared state schema
from agents.state_node import AgentRagState

#import all specialized nodes
from agents.final_response_node import finalResponseAgent
from agents.llm_node import GeneralLLMAgent
from agents.rag_node import RagAgent
from agents.router import RouterAgent
from agents.supervisor_node import SupervisorAgent
from agents.validation_node import ValidationAgent
from agents.validation_router import route_after_validation
from agents.web_node import WebSearchAgent

from langgraph.graph import StateGraph, START, END

logger = setup_logger()

# initializing the state graph with defined State Schema
workflow = StateGraph(AgentRagState)

# registering all the nodes
workflow.add_node("supervisor",SupervisorAgent)
workflow.add_node("rag_agent",RagAgent)
workflow.add_node("general_llm_agent",GeneralLLMAgent)
workflow.add_node("web_search_agent",WebSearchAgent)
workflow.add_node("validator",ValidationAgent)
workflow.add_node("finalize",finalResponseAgent)

# set the inital entry point of the graph
workflow.add_edge(START,"supervisor")

# setting up the supervisor router
workflow.add_conditional_edges(
    "supervisor",
    RouterAgent,
    {
        "rag_agent": "rag_agent",
        "general_llm_agent": "general_llm_agent",
        "web_search_agent": "web_search_agent",
        "end_node": "finalize"
    }
)

# direct all nodes output to validation node
workflow.add_edge("rag_agent","validator")
workflow.add_edge("general_llm_agent","validator")
workflow.add_edge("web_search_agent","validator")

# setting up the validation router
workflow.add_conditional_edges(
    "validator",
    route_after_validation,
    {
        "route_to_finalize": "finalize",
        "route_to_supervisor": "supervisor"
    }
)


workflow.add_edge("finalize",END)

# compliling the application
app = workflow.compile()

logger.info("AI Researcher Assistacne workflow compilied successfully and ready for execution")