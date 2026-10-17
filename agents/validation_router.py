from agents.state_node import AgentRagState
from budget import budget_from_config
from langchain_core.runnables import RunnableConfig
from monitoring.logger import setup_logger


logger = setup_logger()

def route_after_validation(state: AgentRagState, config: RunnableConfig):
    """Finalize on a pass, a stopped run, or when retries/budget are used up.

    Reaching a limit only ends the run; it never turns a failed check into a pass
    (the finalize node labels such answers as not validated / incomplete).
    """
    validation_metadata = state.get("validation_metadata", {}) or {}
    is_valid_signal = str(validation_metadata.get("is_valid", "no")).lower().strip()

    if state.get("termination_reason"):
        logger.warning(f"ValidationRouter | Run stopped ({state['termination_reason']}). Routing to finalize.")
        return "route_to_finalize"

    logger.info(f"Validation Router Status: {is_valid_signal}")

    if is_valid_signal == "yes" and validation_metadata.get("performed"):
        return "route_to_finalize"

    budget = budget_from_config(config)
    reason = budget.exhausted()
    if reason:
        logger.warning(f"ValidationRouter | Budget exhausted ({reason}). Routing to finalize.")
        return "route_to_finalize"
    if budget.retry_rounds_remaining() == 0:
        logger.warning("ValidationRouter | Retry rounds used up. Routing to finalize.")
        return "route_to_finalize"
    return "route_to_supervisor"
