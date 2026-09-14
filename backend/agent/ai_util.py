import asyncio
import contextvars
from collections.abc import Collection

import ai

current_tool_call_id: contextvars.ContextVar[str | None] = contextvars.ContextVar(
    "current_tool_call_id", default=None
)


class TrackingToolRunner(ai.ToolRunner):
    def __init__(self) -> None:
        super().__init__()
        self.id_to_task: dict[str, asyncio.Task[ai.events.ToolCallResult]] = {}
        self.task_to_id: dict[asyncio.Task[ai.events.ToolCallResult], str] = {}

    def discard(self, task: asyncio.Task[ai.events.ToolCallResult]) -> None:
        if task in self.task_to_id:
            del self.id_to_task[self.task_to_id[task]]
            del self.task_to_id[task]
        super().discard(task)

    def discard_except(self, ok_calls: Collection[ai.ToolCall]) -> None:
        ok_ids = {tc.id for tc in ok_calls}
        for id, task in list(self.id_to_task.items()):
            if id not in ok_ids:
                self.discard(task)

    def schedule(
        self, tc: ai.agents.agent.ToolCallCallable
    ) -> asyncio.Task[ai.events.ToolCallResult]:
        id = getattr(tc, "id", None)
        if id and id in self.id_to_task:
            return self.id_to_task[id]

        async def _run() -> ai.events.ToolCallResult:
            if id:
                current_tool_call_id.set(id)
            return await tc()

        task = super().schedule(_run)
        if id:
            self.id_to_task[id] = task
            self.task_to_id[task] = id
        return task
