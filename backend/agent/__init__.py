TASK_QUEUE = "seal-temporal"


def session_workflow_id(session_id: str) -> str:
    return f"seal-session:{session_id}"
