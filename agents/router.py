from agents.state_node import AgentRagState
from monitoring.logger import setup_logger


logger = setup_logger()

def RouterAgent(state:AgentRagState):
   

   """
   It simply reads the path already determined by the Supervisor.
   """

   # Grab the decision straight from the state, fallback if empty
   return state.get("next_step", "general_llm_agent")

