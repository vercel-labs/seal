from __future__ import annotations

import asyncio
import logging
import os

import temporalio.client
import temporalio.worker

from agent import TASK_QUEUE, driver, telemetry, temporal, turn

logging.getLogger("httpx2").setLevel(logging.WARNING)


async def main() -> None:
    telemetry.install("seal-agent")
    client = await temporalio.client.Client.connect(
        os.environ.get("TEMPORAL_ADDRESS", "localhost:7233"),
        namespace=os.environ.get("TEMPORAL_NAMESPACE", "default"),
        data_converter=temporal.DATA_CONVERTER,
    )
    async with temporalio.worker.Worker(
        client,
        task_queue=TASK_QUEUE,
        workflows=[driver.SessionWorkflow, turn.TurnWorkflow],
        activities=turn.ACTIVITIES,
    ):
        print(f"[seal] Temporal worker running on {TASK_QUEUE!r}", flush=True)
        await asyncio.Event().wait()


def run() -> None:
    asyncio.run(main())


if __name__ == "__main__":
    run()
