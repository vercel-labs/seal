import { getToolName, isToolUIPart } from "ai";
import type { UIMessage } from "ai";

/**
 * `sendAutomaticallyWhen` predicate for useChat: resubmit once every approval
 * in the last step is answered and every other gated tool has settled.
 * Subagent calls are exempt — their output arrives out-of-band via the
 * durable driver, so they must not block the approval resubmission.
 */
export function lastAssistantMessageIsCompleteWithSealApprovals({
  messages,
}: {
  messages: UIMessage[];
}): boolean {
  const message = messages[messages.length - 1];

  if (!message || message.role !== "assistant") {
    return false;
  }

  const lastStepStartIndex = message.parts.reduce((lastIndex, part, index) => {
    return part.type === "step-start" ? index : lastIndex;
  }, -1);

  const toolParts = message.parts
    .slice(lastStepStartIndex + 1)
    .filter(isToolUIPart);
  const approvalParts = toolParts.filter((part) => part.approval);

  return (
    approvalParts.length > 0 &&
    approvalParts.every((part) => part.state === "approval-responded") &&
    toolParts.every((part) => {
      if (part.approval || getToolName(part) === "subagent") {
        return true;
      }
      return part.state === "output-available" || part.state === "output-error";
    })
  );
}
