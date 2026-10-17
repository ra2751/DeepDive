from budget.manager import (
    BudgetConfigError,
    BudgetExceeded,
    BudgetLimits,
    BudgetManager,
    UsageLedger,
    format_cost,
    format_usage,
)
from budget.pricing import PriceTable


def budget_from_config(config) -> BudgetManager:
    """Return the run-scoped BudgetManager passed via config['configurable']['budget']."""
    budget = ((config or {}).get("configurable") or {}).get("budget")
    if not isinstance(budget, BudgetManager):
        raise BudgetConfigError(
            "No research budget for this run. Invoke the graph with "
            "config={'configurable': {'budget': BudgetManager(...)}}."
        )
    return budget


def new_research_budget(settings) -> BudgetManager:
    return BudgetManager(
        BudgetLimits.from_settings(settings),
        pricing=PriceTable.load(settings.mesh_pricing_file),
        name="research",
    )


def new_ingestion_budget(settings) -> BudgetManager:
    return BudgetManager(
        BudgetLimits.ingestion_from_settings(settings),
        pricing=PriceTable.load(settings.mesh_pricing_file),
        name="ingestion",
    )


__all__ = [
    "BudgetConfigError", "BudgetExceeded", "BudgetLimits", "BudgetManager", "UsageLedger",
    "PriceTable", "format_cost", "format_usage", "budget_from_config",
    "new_research_budget", "new_ingestion_budget",
]
