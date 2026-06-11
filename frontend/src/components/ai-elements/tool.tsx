"use client";

import type { DynamicToolUIPart, ToolUIPart } from "ai";
import type { ComponentProps, ReactNode } from "react";

import {
  Collapsible,
  CollapsibleContent,
  CollapsibleTrigger,
} from "@/components/ui/collapsible";
import { cn } from "@/lib/utils";
import { CheckIcon, XIcon } from "lucide-react";
import { isValidElement } from "react";

// tool payloads are shown one object-level deep: dimmed `key:` labels with the
// raw value (no JSON quoting/escaping), nested values as compact JSON
const formatValue = (value: unknown): string =>
  typeof value === "string" ? value : JSON.stringify(value);

const asEntries = (value: unknown): Record<string, unknown> | null =>
  value && typeof value === "object" && !Array.isArray(value)
    ? (value as Record<string, unknown>)
    : null;

const ObjectEntries = ({ data }: { data: Record<string, unknown> }) => (
  <div className="space-y-1">
    {Object.entries(data).map(([key, value]) => (
      <div key={key} className="flex min-w-0 gap-2">
        <span className="shrink-0 text-muted-foreground">{key}:</span>
        <span className="min-w-0 whitespace-pre-wrap break-words">
          {formatValue(value)}
        </span>
      </div>
    ))}
  </div>
);

export type ToolProps = ComponentProps<typeof Collapsible>;

export const Tool = ({ className, ...props }: ToolProps) => (
  <Collapsible
    className={cn("group not-prose mb-4 w-full", className)}
    {...props}
  />
);

export type ToolPart = ToolUIPart | DynamicToolUIPart;

export type ToolHeaderProps = {
  title?: string;
  className?: string;
  input?: ToolPart["input"];
  onApprovalResponse?: (approved: boolean) => void;
} & (
  | { type: ToolUIPart["type"]; state: ToolUIPart["state"]; toolName?: never }
  | {
      type: DynamicToolUIPart["type"];
      state: DynamicToolUIPart["state"];
      toolName: string;
    }
);

const statusLabels: Record<ToolPart["state"], string> = {
  "approval-requested": "Awaiting Approval",
  "approval-responded": "Responded",
  "input-available": "Running",
  "input-streaming": "Pending",
  "output-available": "Completed",
  "output-denied": "Denied",
  "output-error": "Error",
};

// monochrome glyphs: ● done, ○ in flight (pulsing while active), ! attention
const statusGlyphs: Record<
  ToolPart["state"],
  { glyph: string; className: string }
> = {
  "approval-requested": {
    glyph: "!",
    className: "animate-pulse text-foreground",
  },
  "approval-responded": {
    glyph: "○",
    className: "animate-pulse text-muted-foreground",
  },
  "input-available": {
    glyph: "○",
    className: "animate-pulse text-muted-foreground",
  },
  "input-streaming": { glyph: "○", className: "text-muted-foreground/50" },
  "output-available": { glyph: "●", className: "text-muted-foreground" },
  "output-denied": { glyph: "○", className: "text-muted-foreground/50" },
  "output-error": { glyph: "!", className: "text-destructive" },
};

export const ToolHeader = ({
  className,
  title,
  type,
  state,
  toolName,
  input,
  onApprovalResponse,
  ...props
}: ToolHeaderProps) => {
  const derivedName =
    type === "dynamic-tool" ? toolName : type.split("-").slice(1).join("-");
  const status = statusGlyphs[state];

  return (
    <div className={cn("flex w-full items-center gap-2", className)}>
      <CollapsibleTrigger
        className="flex min-w-0 flex-1 items-center gap-2 py-1 text-left"
        {...props}
      >
        <span
          // w-4 + gap-2 = 24px, so the glyph sits in the same gutter as the
          // message symbols and the tool name starts at the shared text column
          className={cn("w-4 shrink-0 text-left text-xs", status.className)}
          title={statusLabels[state]}
        >
          {status.glyph}
        </span>
        <span className="shrink-0 text-sm">{title ?? derivedName}</span>
        {input != null && (
          <span className="min-w-0 truncate text-muted-foreground/60 text-xs group-data-[state=open]:hidden">
            {asEntries(input)
              ? Object.entries(asEntries(input)!)
                  .map(([key, value]) => `${key}: ${formatValue(value)}`)
                  .join("  ")
              : formatValue(input)}
          </span>
        )}
        <span className="sr-only">{statusLabels[state]}</span>
      </CollapsibleTrigger>
      {state === "approval-requested" && onApprovalResponse && (
        <span className="flex shrink-0 items-center gap-1">
          <button
            type="button"
            aria-label="Reject"
            className="p-1 text-muted-foreground hover:text-foreground"
            onClick={() => onApprovalResponse(false)}
          >
            <XIcon className="size-3.5" />
          </button>
          <button
            type="button"
            aria-label="Approve"
            className="p-1 text-muted-foreground hover:text-foreground"
            onClick={() => onApprovalResponse(true)}
          >
            <CheckIcon className="size-3.5" />
          </button>
        </span>
      )}
    </div>
  );
};

export type ToolContentProps = ComponentProps<typeof CollapsibleContent>;

export const ToolContent = ({ className, ...props }: ToolContentProps) => (
  <CollapsibleContent
    className={cn(
      // the left rule hangs from the status glyph (left-aligned text-xs char,
      // center ~3px; ml-[3px] + 1px border lines up underneath it)
      "data-[state=closed]:fade-out-0 data-[state=closed]:slide-out-to-top-2 data-[state=open]:slide-in-from-top-2 ml-[3px] space-y-3 border-l py-1 pl-4 text-popover-foreground outline-none data-[state=closed]:animate-out data-[state=open]:animate-in",
      className,
    )}
    {...props}
  />
);

export type ToolInputProps = ComponentProps<"div"> & {
  input: ToolPart["input"];
};

export const ToolInput = ({ className, input, ...props }: ToolInputProps) => {
  if (input == null) return null;

  const entries = asEntries(input);

  return (
    <div
      className={cn("space-y-1 overflow-hidden text-xs", className)}
      {...props}
    >
      <h4 className="text-[10px] text-muted-foreground/50 uppercase tracking-wide">
        Parameters
      </h4>
      {entries ? (
        <ObjectEntries data={entries} />
      ) : (
        <div className="whitespace-pre-wrap break-words">
          {formatValue(input)}
        </div>
      )}
    </div>
  );
};

export type ToolOutputProps = ComponentProps<"div"> & {
  output: ToolPart["output"];
  errorText: ToolPart["errorText"];
};

export const ToolOutput = ({
  className,
  output,
  errorText,
  ...props
}: ToolOutputProps) => {
  if (!(output || errorText)) {
    return null;
  }

  let Output = <div>{output as ReactNode}</div>;

  if (typeof output === "object" && !isValidElement(output)) {
    const entries = asEntries(output);
    Output = entries ? (
      <ObjectEntries data={entries} />
    ) : (
      <div className="whitespace-pre-wrap break-words">
        {JSON.stringify(output, null, 2)}
      </div>
    );
  } else if (typeof output === "string") {
    Output = <div className="whitespace-pre-wrap break-words">{output}</div>;
  }

  return (
    <div className={cn("space-y-1", className)} {...props}>
      <h4 className="text-[10px] text-muted-foreground/50 uppercase tracking-wide">
        {errorText ? "Error" : "Result"}
      </h4>
      <div
        className={cn(
          "overflow-x-auto text-xs [&_table]:w-full",
          errorText ? "text-destructive" : "text-foreground",
        )}
      >
        {errorText && <div>{errorText}</div>}
        {Output}
      </div>
    </div>
  );
};
