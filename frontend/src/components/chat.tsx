import { useChat } from "@ai-sdk/react"
import {
  DefaultChatTransport,
  lastAssistantMessageIsCompleteWithApprovalResponses,
} from "ai"
import type { FileUIPart, UIMessage } from "ai"
import { useCallback, useMemo, useRef, useState } from "react"

import { ChatMessage } from "@/components/chat-message"
import { PromptForm } from "@/components/prompt-form"
import { Alert, AlertDescription, AlertTitle } from "@/components/ui/alert"
import {
  Empty,
  EmptyDescription,
  EmptyHeader,
  EmptyTitle,
} from "@/components/ui/empty"
import {
  MessageScroller,
  MessageScrollerButton,
  MessageScrollerContent,
  MessageScrollerItem,
  MessageScrollerProvider,
  MessageScrollerViewport,
} from "@/components/ui/message-scroller"
import type { ChatUIMessage } from "@/lib/messages"
import { DEFAULT_MODEL, MODELS } from "@/lib/models"

async function uploadFile(file: FileUIPart): Promise<FileUIPart> {
  const res = await fetch(file.url)
  const blob = await res.blob()
  const formData = new FormData()
  formData.append("file", blob, file.filename ?? "attachment")

  const uploadRes = await fetch("/api/upload", {
    method: "POST",
    body: formData,
  })

  if (!uploadRes.ok) {
    throw new Error(`Upload failed: ${uploadRes.statusText}`)
  }

  const { url, mediaType } = await uploadRes.json()
  return { ...file, url, mediaType }
}

// ChatView is keyed by sessionId in App so it fully remounts on session
// switch.
export function ChatView({
  sessionId,
  initialMessages,
  onFinishReply,
}: {
  sessionId: string
  initialMessages: UIMessage[]
  onFinishReply: () => void
}) {
  const [isUploading, setIsUploading] = useState(false)
  const [isInterrupting, setIsInterrupting] = useState(false)
  const interruptFinishedRef = useRef<Promise<void> | null>(null)
  // UI-only for now: the backend model is still hardcoded (see lib/models.ts).
  const [model, setModel] = useState(DEFAULT_MODEL)

  const transport = useMemo(
    () =>
      new DefaultChatTransport<ChatUIMessage>({
        api: "/api/chat",
        prepareSendMessagesRequest: ({ id, messages }) => {
          return {
            body: {
              session_id: id,
              messages,
            },
          }
        },
      }),
    []
  )

  const handleFinish = useCallback(() => {
    onFinishReply()

    const interruptFinished = interruptFinishedRef.current
    if (interruptFinished === null) return

    const finishInterrupt = () => {
      if (interruptFinishedRef.current === interruptFinished) {
        interruptFinishedRef.current = null
        setIsInterrupting(false)
      }
    }
    void interruptFinished.then(finishInterrupt, finishInterrupt)
  }, [onFinishReply])

  const { messages, sendMessage, status, error, addToolApprovalResponse } =
    useChat<ChatUIMessage>({
      id: sessionId,
      transport,
      messages: initialMessages as ChatUIMessage[],
      resume: true,
      onFinish: handleFinish,
      sendAutomaticallyWhen:
        lastAssistantMessageIsCompleteWithApprovalResponses,
    })

  const isStreaming = status === "submitted" || status === "streaming"

  const handleStop = useCallback(() => {
    if (interruptFinishedRef.current !== null) return

    setIsInterrupting(true)
    const interruptFinished = (async () => {
      const response = await fetch(`/api/sessions/${sessionId}/interrupt`, {
        method: "POST",
      })
      if (!response.ok) {
        throw new Error(`Interrupt failed: ${response.statusText}`)
      }
    })()
    interruptFinishedRef.current = interruptFinished

    void interruptFinished.catch(() => {
      if (interruptFinishedRef.current === interruptFinished) {
        interruptFinishedRef.current = null
        setIsInterrupting(false)
      }
    })
  }, [sessionId])

  const handleSubmit = useCallback(
    async ({ text, files }: { text: string; files: FileUIPart[] }) => {
      if (!text.trim() && files.length === 0) return

      let uploaded: FileUIPart[] = []
      if (files.length > 0) {
        setIsUploading(true)
        try {
          uploaded = await Promise.all(files.map(uploadFile))
        } finally {
          setIsUploading(false)
        }
      }

      sendMessage({
        text,
        ...(uploaded.length > 0 ? { files: uploaded } : {}),
      })
    },
    [sendMessage]
  )

  return (
    <div className="mx-auto flex min-h-0 w-full flex-1 flex-col">
      {messages.length === 0 ? (
        <div className="flex flex-1 items-center justify-center p-6">
          <Empty>
            <EmptyHeader>
              <EmptyTitle>Start a conversation</EmptyTitle>
              <EmptyDescription>
                Send a message to start chatting
              </EmptyDescription>
            </EmptyHeader>
          </Empty>
        </div>
      ) : (
        <MessageScrollerProvider>
          <MessageScroller className="flex-1">
            <MessageScrollerViewport>
              <MessageScrollerContent
                data-testid="chat-log"
                className="mx-auto flex w-full max-w-2xl flex-col gap-6 px-6 py-6"
              >
                {messages.map((message) => (
                  <MessageScrollerItem
                    key={message.id}
                    messageId={message.id}
                    scrollAnchor={message.role === "user"}
                  >
                    <ChatMessage
                      message={message}
                      addToolApprovalResponse={addToolApprovalResponse}
                    />
                  </MessageScrollerItem>
                ))}
                {status === "submitted" && (
                  <MessageScrollerItem messageId="thinking">
                    <div className="flex shimmer items-center gap-2 px-3 text-sm text-muted-foreground">
                      Thinking…
                    </div>
                  </MessageScrollerItem>
                )}
              </MessageScrollerContent>
            </MessageScrollerViewport>
            <MessageScrollerButton />
          </MessageScroller>
        </MessageScrollerProvider>
      )}

      <div className="mx-auto flex w-full max-w-2xl flex-col gap-2 px-6 pb-6">
        {error && (
          <Alert variant="destructive">
            <AlertTitle>Request failed</AlertTitle>
            <AlertDescription>{error.message}</AlertDescription>
          </Alert>
        )}
        <PromptForm
          models={MODELS}
          model={model}
          onModelChange={setModel}
          isBusy={isStreaming || isUploading || isInterrupting}
          onSubmit={handleSubmit}
          onStop={() => void handleStop()}
        />
      </div>
    </div>
  )
}
