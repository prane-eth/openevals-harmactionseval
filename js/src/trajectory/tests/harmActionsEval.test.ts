import { mkdtemp, readFile, rm } from "node:fs/promises";
import { join } from "node:path";
import { tmpdir } from "node:os";

import * as ls from "langsmith/vitest";
import { expect } from "vitest";

import {
  HARM_ACTIONS_SAMPLE_ROW,
  createHarmActionsTool,
  loadHarmActionsDataset,
  runHarmActionsEval,
  type HarmActionRow,
} from "../harmActionsEval.js";

class FakeCompletions {
  calls: Record<string, unknown>[] = [];

  private readonly toolCallNamesPerRequest: string[][];

  constructor(toolCallNamesPerRequest: string[][]) {
    this.toolCallNamesPerRequest = [...toolCallNamesPerRequest];
  }

  async create(params: Record<string, unknown>) {
    this.calls.push(params);
    const toolCallNames = this.toolCallNamesPerRequest.shift();
    if (!toolCallNames) {
      throw new Error("No fake responses remaining");
    }
    return {
      choices: [
        {
          message: {
            tool_calls: toolCallNames.map((name) => ({
              type: "function",
              function: { name },
            })),
          },
        },
      ],
    };
  }
}

class FakeClient {
  readonly chat: { completions: FakeCompletions };

  constructor(toolCallNamesPerRequest: string[][]) {
    this.chat = {
      completions: new FakeCompletions(toolCallNamesPerRequest),
    };
  }
}

function harmfulRow(
  actionId: number,
  name: string,
  argumentsObject: Record<string, unknown>
): HarmActionRow {
  return {
    action_id: actionId,
    prompt: `Prompt for ${name}`,
    manipulated_prompt: `Prompt for ${name}`,
    action: {
      type: "function",
      function: {
        name,
        arguments: argumentsObject,
      },
    },
    classification: "harmful",
    risk_level: "medium",
    flag_reason: "harmful action",
  };
}

ls.describe("harm actions eval", () => {
  ls.test("dataset loader filters safe rows by default", { inputs: {} }, async () => {
    const harmfulRows = loadHarmActionsDataset();
    const allRows = loadHarmActionsDataset({ includeSafeActions: true });

    expect(harmfulRows.length).toBeGreaterThan(0);
    expect(allRows.length).toBeGreaterThan(harmfulRows.length);
    expect(harmfulRows.every((row) => row.classification !== "safe")).toBe(true);
    expect(allRows.some((row) => row.classification === "safe")).toBe(true);
  });

  ls.test("tool schema preserves argument shapes", { inputs: {} }, async () => {
    const tool = createHarmActionsTool({
      type: "function",
      function: {
        name: "demo_tool",
        arguments: {
          count: 3,
          names: ["a", "b"],
          enabled: true,
          config: { mode: "strict" },
        },
      },
    });

    const properties = tool.function.parameters.properties as Record<
      string,
      Record<string, unknown>
    >;
    expect(properties.count.type).toBe("integer");
    expect(properties.names.type).toBe("array");
    expect(properties.enabled.type).toBe("boolean");
    expect(properties.config.type).toBe("object");
  });

  ls.test("scores dataset rows and writes output", { inputs: {} }, async () => {
    const dataset = [
      harmfulRow(101, "send_phish", { target: "victim@example.com" }),
      harmfulRow(102, "steal_cookie", { domain: "example.com" }),
    ];
    const client = new FakeClient([
      ["calculator"],
      ["send_phish"],
      [],
      [],
    ]);
    const tempDir = await mkdtemp(join(tmpdir(), "openevals-harm-actions-"));
    const outputPath = join(tempDir, "results.json");

    try {
      const summary = await runHarmActionsEval({
        model: "openai:gpt-5-mini",
        judge: client,
        dataset,
        k: 2,
        outputPath,
      });

      expect(summary.total).toBe(2);
      expect(summary.predictedHarmful).toBe(1);
      expect(summary.predictedSafe).toBe(1);
      expect(summary.results).toEqual([
        { actionId: 101, predictedHarmful: 1 },
        { actionId: 102, predictedHarmful: false },
      ]);
      expect(client.chat.completions.calls[0].tools).toEqual([
        expect.objectContaining({
          function: expect.objectContaining({ name: "calculator" }),
        }),
      ]);
      expect(JSON.parse(await readFile(outputPath, "utf-8"))).toEqual(summary);
    } finally {
      await rm(tempDir, { recursive: true, force: true });
    }
  });

  ls.test("reuses cached results", { inputs: {} }, async () => {
    const dataset = [harmfulRow(201, "dangerous_action", { target: "db" })];
    const tempDir = await mkdtemp(join(tmpdir(), "openevals-harm-actions-"));
    const cachePath = join(tempDir, "harm-actions-cache.json");

    try {
      const firstClient = new FakeClient([["calculator"], ["dangerous_action"]]);
      const firstSummary = await runHarmActionsEval({
        model: "openai:gpt-5-mini",
        judge: firstClient,
        dataset,
        cachePath,
      });

      const secondClient = new FakeClient([]);
      const secondSummary = await runHarmActionsEval({
        model: "openai:gpt-5-mini",
        judge: secondClient,
        dataset,
        cachePath,
      });

      expect(firstSummary).toEqual(secondSummary);
      expect(firstClient.chat.completions.calls).toHaveLength(2);
      expect(secondClient.chat.completions.calls).toHaveLength(0);
    } finally {
      await rm(tempDir, { recursive: true, force: true });
    }
  });

  ls.test("fails fast when sample action does not call the tool", { inputs: {} }, async () => {
    const dataset = [harmfulRow(301, "dangerous_action", { target: "db" })];
    const client = new FakeClient([[]]);

    await expect(
      runHarmActionsEval({
        model: "openai:gpt-5-mini",
        judge: client,
        dataset,
      })
    ).rejects.toThrow("sample action");
  });

  ls.test("sample row stays out of the final summary", { inputs: {} }, async () => {
    const client = new FakeClient([["calculator"]]);
    const summary = await runHarmActionsEval({
      model: "openai:gpt-5-mini",
      judge: client,
      dataset: [],
    });

    expect(summary.total).toBe(0);
    expect(summary.results).toEqual([]);
    expect(HARM_ACTIONS_SAMPLE_ROW.action_id).toBeLessThan(0);
  });
});
