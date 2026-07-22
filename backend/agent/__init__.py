import os

import vercel.workflow

workflow = vercel.workflow.Workflows(
    sandbox_policy=vercel.workflow.SandboxPolicy(
        passthrough_modules=frozenset(
            {
                "rich",  # annoying terminal detection stuff
                "modelsdotdev",  # sqlite database
            }
            | (
                {"ai"}
                if os.environ.get("SEAL_NO_PASSTHROUGH_AI", "1") == "0"
                else set()
            )
        ),
        cleanups=vercel.workflow.sandbox.ALL_CLEANUPS,
    )
)
