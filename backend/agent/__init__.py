import vercel.workflow

# app must have one registry shared across all workflows.
# this is because the handler matches messages by prefix __wkf_*
# and nothing else. if there's two registries, one of them will pick up
# other one's message (like __wkf_step_...) and raise a KeyError because
# it doesn't know what to do with it.
workflow = vercel.workflow.Workflows(
    sandbox_policy=vercel.workflow.SandboxPolicy(
        passthrough_modules=frozenset(
            {
                "ai.telemetry",  # needs the time
                "rich",  # annoying terminal detection stuff
                "modelsdotdev",  # sqlite database
            }
        ),
        cleanups=vercel.workflow.sandbox.ALL_CLEANUPS,
    )
)
