import type { UIMessage } from "ai"

// seal's tools are defined in the Python backend, so the tool map is written
// by hand (InferUITools needs TypeScript tool definitions). Tools that are
// missing here still arrive at runtime as the backend grows new ones — they
// render through the generic ToolPart fallback.
export type SealTools = {
  subagent: {
    input: { prompt?: string; name?: string | null }
    output: unknown
  }
  generate_image: { input: { prompt?: string }; output: unknown }
  bash: {
    input: { command?: string; timeout?: number | null }
    output: unknown
  }
  web_fetch: {
    input: { url?: string; method?: string; headers?: string; body?: string }
    output: unknown
  }
}

export type ChatUIMessage = UIMessage<unknown, Record<string, never>, SealTools>

export type ChatMessagePart = ChatUIMessage["parts"][number]

export type ChatToolPart = Extract<ChatMessagePart, { toolCallId: string }>

export type SubagentToolPart = Extract<
  ChatMessagePart,
  { type: "tool-subagent" }
>
export type GenerateImageToolPart = Extract<
  ChatMessagePart,
  { type: "tool-generate_image" }
>
export type BashToolPart = Extract<ChatMessagePart, { type: "tool-bash" }>
export type WebFetchToolPart = Extract<
  ChatMessagePart,
  { type: "tool-web_fetch" }
>
