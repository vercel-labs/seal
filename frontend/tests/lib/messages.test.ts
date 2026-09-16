import { describe, expect, it } from "vitest"

import { getFreshParts } from "../../src/lib/messages"

const text = (i: number) => ({ type: "text", i })

describe("getFreshParts", () => {
  it("drops the previous step on data-reload", () => {
    expect(
      getFreshParts([
        { type: "step-start" },
        text(1),
        { type: "data-reload" },
        text(2),
      ])
    ).toEqual([{ type: "step-start" }, text(2)])
  })

  it("drops everything on data-reload when there is no step boundary", () => {
    // seeded reload history carries no step-start parts; the replayed turn
    // replaces it wholesale.
    expect(getFreshParts([text(1), { type: "data-reload" }, text(2)])).toEqual([
      text(2),
    ])
  })

  it("leaves ordinary parts unchanged", () => {
    const parts = [
      text(1),
      { type: "tool-bash", toolCallId: "tc-a" },
      { type: "step-start" },
      { type: "tool-bash", toolCallId: "tc-a" },
    ]
    expect(getFreshParts(parts)).toEqual(parts)
  })
})
