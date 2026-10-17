from agents.state_node import AgentRagState
from budget import budget_from_config
from langchain_core.messages import AIMessage
from langchain_core.runnables import RunnableConfig
from monitoring.logger import setup_logger

logger = setup_logger()

# Limits whose exhaustion makes a run "incomplete" rather than "not validated"
BUDGET_REASONS = {
    "max_llm_calls", "max_embedding_calls", "max_web_searches", "max_tool_calls",
    "max_run_seconds", "max_mesh_cost_usd", "mesh_spend_limit",
}

STATUS_TEXT = {
    "answer_check_passed": "Answer check passed (answer-level check against the retrieved sources; not claim-level verification)",
    "ungrounded": "Ungrounded: no sources were used, so this answer could not be checked against evidence",
    "not_validated": "Not validated: the answer did not pass the answer-level check",
    "incomplete_budget": "Incomplete: the run stopped at a budget or time limit before the answer passed the check",
    "out_of_scope": "Out of scope",
}


def derive_answer_status(state, budget):
    """Return (answer_status, termination_reason). A status is never upgraded by a limit."""
    reason = state.get("termination_reason")
    if state.get("out_of_scope"):
        return "out_of_scope", reason
    vm = state.get("validation_metadata") or {}
    passed = (
        str(vm.get("is_valid", "no")).lower().strip() == "yes"
        and vm.get("performed") is True
        and not reason
    )
    if passed and state.get("draft_answer"):
        return ("answer_check_passed" if state.get("retrieved_docs") else "ungrounded"), None
    if reason is None:
        reason = budget.exhausted() or ("max_retry_rounds" if budget.retry_rounds_remaining() == 0 else "validation_failed")
    if reason in BUDGET_REASONS:
        return "incomplete_budget", reason
    return "not_validated", reason


def finalResponseAgent(state:AgentRagState, config: RunnableConfig):
    """
    The FinalResponseAgent takes the latest draft answer and the validation results and
    produces the final response with an honest status label (answer_status).
    """
    budget = budget_from_config(config)
    status, reason = derive_answer_status(state, budget)

    final_answer = state.get("draft_answer")
    if not final_answer:
        final_answer = f"No answer could be produced ({reason or 'unknown reason'})."

    logger.info(f"FinalResponseNode| status={status} | termination_reason={reason}")

    return {
        "messages": [AIMessage(content=final_answer)],
        "final_response": final_answer,
        "answer_status": status,
        "termination_reason": reason,
        "usage": budget.snapshot().model_dump(),
        "next_step": "",
        "loop_count": 0
    }
