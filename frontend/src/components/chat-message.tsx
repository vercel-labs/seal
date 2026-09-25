import { getToolName, isToolUIPart } from "ai"
import type { ChatAddToolApproveResponseFunction, UIMessage } from "ai"
import { FileIcon } from "lucide-react"
import { Fragment } from "react"
import type { ReactNode } from "react"

import { BashPart } from "@/components/parts/bash-part"
import { GenerateImagePart } from "@/components/parts/generate-image-part"
import { SubagentPart } from "@/components/parts/subagent-part"
import { TextPart } from "@/components/parts/text-part"
import { ToolPart } from "@/components/parts/tool-part"
import { WebFetchPart } from "@/components/parts/web-fetch-part"
import {
  Attachment,
  AttachmentContent,
  AttachmentGroup,
  AttachmentMedia,
  AttachmentTitle,
} from "@/components/ui/attachment"
import { Bubble, BubbleContent } from "@/components/ui/bubble"
import { Message, MessageContent } from "@/components/ui/message"
import type { ChatMessagePart, ChatUIMessage } from "@/lib/messages"

function FileAttachment({
  url,
  mediaType,
  filename,
}: {
  url: string
  mediaType?: string
  filename?: string
}) {
  return (
    <Attachment size="sm">
      {mediaType?.startsWith("image/") ? (
        <AttachmentMedia variant="image">
          <img src={url} alt={filename ?? "attachment"} />
        </AttachmentMedia>
      ) : (
        <AttachmentMedia>
          <FileIcon />
        </AttachmentMedia>
      )}
      <AttachmentContent>
        <AttachmentTitle>{filename ?? "attachment"}</AttachmentTitle>
      </AttachmentContent>
    </Attachment>
  )
}

// A subagent's output is a nested UIMessage streamed live by the durable
// backend — or an array of them when the child agent produced more than one
// bubble. The final/reload path may instead carry a plain-text summary, which
// SubagentPart renders itself.
function subagentMessages(output: unknown): ChatUIMessage[] {
  const isMessage = (value: unknown): value is ChatUIMessage =>
    value != null && typeof value === "object" && "parts" in value
  if (Array.isArray(output)) return output.filter(isMessage)
  return isMessage(output) ? [output] : []
}

// Template-style per-tool dispatch (chatbot-template chat-message.tsx). The
// wrapper div in renderParts carries the data-* attributes the e2e suite
// targets, so the part components stay presentation-only.
function renderToolPart(
  part: Extract<ChatMessagePart, { toolCallId: string }>,
  depth: number,
  addToolApprovalResponse: ChatAddToolApproveResponseFunction
): ReactNode {
  switch (part.type) {
    case "tool-bash":
      return (
        <BashPart part={part} onApprovalResponse={addToolApprovalResponse} />
      )
    case "tool-web_fetch":
      return <WebFetchPart part={part} />
    case "tool-generate_image":
      return <GenerateImagePart part={part} />
    case "tool-subagent": {
      const nested = subagentMessages(part.output)
      return (
        <SubagentPart part={part}>
          {nested.length > 0
            ? nested.map((message, index) => (
                <Fragment key={index}>
                  {renderParts(
                    message.parts,
                    message.role,
                    depth + 1,
                    addToolApprovalResponse
                  )}
                </Fragment>
              ))
            : undefined}
        </SubagentPart>
      )
    }
    default:
      return (
        <ToolPart part={part} onApprovalResponse={addToolApprovalResponse} />
      )
  }
}

// renderParts recurses into subagent tool output; nested content renders at
// depth + 1 so the e2e depth-0 selectors don't match it.
function renderParts(
  parts: ChatMessagePart[],
  role: UIMessage["role"],
  depth: number,
  addToolApprovalResponse: ChatAddToolApproveResponseFunction
): ReactNode {
  return parts.map((part, index) => {
    if (isToolUIPart(part)) {
      return (
        <div
          key={index}
          data-testid="tool-card"
          data-tool-depth={depth}
          data-tool-name={getToolName(part)}
          data-tool-state={part.state}
          className="flex w-full min-w-0 flex-col gap-1.5 py-0.5"
        >
          {renderToolPart(part, depth, addToolApprovalResponse)}
        </div>
      )
    }

    if (part.type === "text") {
      return <TextPart key={index} text={part.text} role={role} depth={depth} />
    }

    if (part.type === "file") {
      return (
        <FileAttachment
          key={index}
          url={part.url}
          mediaType={part.mediaType}
          filename={part.filename}
        />
      )
    }

    return null
  })
}

export function ChatMessage({
  message,
  addToolApprovalResponse,
}: {
  message: ChatUIMessage
  addToolApprovalResponse: ChatAddToolApproveResponseFunction
}) {
  const parts = message.parts

  if (message.role === "user") {
    const text = parts
      .filter((part) => part.type === "text")
      .map((part) => part.text)
      .join("")
    const files = parts.filter((part) => part.type === "file")

    return (
      <Message align="end">
        <MessageContent>
          {files.length > 0 && (
            <AttachmentGroup>
              {files.map((file, index) => (
                <FileAttachment
                  key={index}
                  url={file.url}
                  mediaType={file.mediaType}
                  filename={file.filename}
                />
              ))}
            </AttachmentGroup>
          )}
          {text.trim() && (
            <Bubble
              align="end"
              variant="muted"
              data-testid="message"
              data-message-role="user"
              data-message-depth={0}
            >
              <BubbleContent>{text}</BubbleContent>
            </Bubble>
          )}
        </MessageContent>
      </Message>
    )
  }

  return (
    <Message align="start">
      <MessageContent>
        {renderParts(parts, message.role, 0, addToolApprovalResponse)}
      </MessageContent>
    </Message>
  )
}
