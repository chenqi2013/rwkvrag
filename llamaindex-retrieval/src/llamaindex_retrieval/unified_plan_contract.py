"""Invocation-local Plan grammar bounds, not query generation or semantic repair."""
from copy import deepcopy

from .unified_contracts import Plan, unique


def bounded_plan_schema(task, channels, *, remaining_actions, remaining_web, read_units):
    """Narrow the static schema using only admitted IDs, permissions and budgets.

    Cross-item constraints (duplicates, planned/unplanned overlap, and the Web
    count in a mixed-channel array) still require Plan.check. No unsupported
    uniqueItems/contains/not constraints or silently weakened validation.
    """
    for value in (remaining_actions, remaining_web, read_units):
        if type(value) is not int or value < 0:
            raise ValueError("invalid_plan_budget")
    unique(channels)
    if not set(channels) <= {"knowledge_base", "web"}:
        raise ValueError("invalid_plan_channel")
    goals = [goal.id for goal in task.goals]
    unique(goals)
    if not goals:
        raise ValueError("missing_plan_goals")
    schema = deepcopy(Plan.model_json_schema())
    available = [c for c in channels if c != "web" or remaining_web > 0]
    actions = schema["properties"]["actions"]
    actions["maxItems"] = min(actions["maxItems"], remaining_actions, read_units)
    if not available:
        actions["maxItems"] = 0
    elif available == ["web"]:
        actions["maxItems"] = min(actions["maxItems"], remaining_web)
    action = schema["$defs"]["Action"]["properties"]
    if available:
        action["channel"]["enum"] = available
    # With no available channel, maxItems=0 excludes all actions; keep the
    # unreachable item definition valid instead of emitting an empty enum.
    for field in (action["goal_ids"], schema["properties"]["unplanned_goal_ids"]):
        field["maxItems"] = min(field["maxItems"], len(goals))
        field["items"]["enum"] = list(goals)
    return schema
