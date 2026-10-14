from agents.state_node import AgentRagState
from monitoring.logger import setup_logger


logger = setup_logger()

def route_after_validation(state: AgentRagState):
    #validation_result = state.get("validation_metadata", {}).get("is_valid", "no").lower()
    validation_metadata = state.get("validation_metadata", {})
    reasoning_string = validation_metadata.get("reasoning","")
    is_valid_signal = str(validation_metadata.get("is_valid", "yes")).lower().strip()

    if "CIRCUIT_BREAKER_BYPASS" in reasoning_string:
        logger.warning("ValidationRouter | Circuit breaker bypass detected. Forcing route directly to end_node.")
        return "end_note"

    logger.info(f"Validation Router Status: {is_valid_signal}")

    if is_valid_signal == "yes":
        return "route_to_finalize"
    else:
        return "route_to_supervisor"