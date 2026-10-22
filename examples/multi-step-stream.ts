/**
 * A multi-step `streamText` run, where a later step fails.
 *
 * Step 1 streams a tool call, the SDK runs the tool, and step 2 asks the model
 * for the answer. Step 2 fails with HTTP 529.
 *
 * - The retryable call alone cannot recover it. The tool call of step 1 is
 *   content, so the attempt committed before step 2 started. The error goes to
 *   the stream.
 * - A retryable model recovers it. The SDK calls `doStream` one time for each
 *   step, so the retryable model retries step 2 alone, on its fallback model.
 *   Step 1 and the tool result stay.
 *
 * Offline: all models are mocks, no API key needed.
 *
 * Run:
 *   pnpm build && pnpm tsx examples/multi-step-stream.ts
 *
 * Output:
 *   retryable call only:
 *     steps: anthropic/claude-sonnet-4-5 -> tool-calls
 *     text:  ""
 *     error: Overloaded
 *   retryable call + retryable model:
 *     steps: anthropic/claude-sonnet-4-5 -> tool-calls, openai/gpt-4o -> stop
 *     text:  "It is sunny in Berlin, 24°C."
 *     error: none
 */
import { APICallError, isStepCount, tool } from 'ai';
import { createRetryableModel, httpStatus } from 'ai-retry/language-model';
import { retryableStreamText } from 'ai-retry/stream-text';
import { timeout } from 'ai-retry/stream-text/conditions';
import { Language, MockLanguageModel } from 'ai-test-kit/language';
import { z } from 'zod';

const overloaded = () =>
  new APICallError({
    message: 'Overloaded',
    statusCode: 529,
    url: 'https://api.example.com/v1/messages',
    requestBodyValues: {},
    isRetryable: true,
  });

/** Step 1 calls the weather tool. Step 2 fails with HTTP 529. */
const createPrimaryModel = () =>
  MockLanguageModel.from(
    [
      {
        doStream: [
          Language.streamStart(),
          ...Language.streamToolInput({
            id: 'call-1',
            toolName: 'weather',
            input: { city: 'Berlin' },
          }),
          Language.toolCall({
            toolCallId: 'call-1',
            toolName: 'weather',
            input: { city: 'Berlin' },
          }),
          Language.streamFinish({ finishReason: 'tool-calls' }),
        ],
      },
      overloaded(),
    ],
    { provider: 'anthropic', modelId: 'claude-sonnet-4-5' },
  );

const fallbackModel = MockLanguageModel.from('It is sunny in Berlin, 24°C.', {
  provider: 'openai',
  modelId: 'gpt-4o',
});

const tools = {
  weather: tool({
    description: 'Get the weather for a city',
    inputSchema: z.object({ city: z.string() }),
    execute: async ({ city }) => ({ city, condition: 'sunny', celsius: 24 }),
  }),
};

async function run(
  label: string,
  model: Parameters<typeof retryableStreamText>[0]['model'],
) {
  const result = await retryableStreamText({
    model,
    tools,
    stopWhen: isStepCount(3),
    prompt: 'What is the weather in Berlin?',
    timeout: { firstChunkMs: 2_000 },
    retry: [timeout().switch({ model: fallbackModel })],
    /** Keeps the SDK from logging the stream error, which the run prints itself. */
    onError: () => {},
  });

  let text = '';
  let streamError: unknown;
  for await (const part of result.fullStream) {
    if (part.type === 'text-delta') text += part.text;
    if (part.type === 'error') streamError = part.error;
  }

  const steps = await result.steps;
  console.log(`${label}:`);
  console.log(
    `  steps: ${steps.map((step) => `${step.model.provider}/${step.model.modelId} -> ${step.finishReason}`).join(', ')}`,
  );
  console.log(`  text:  ${JSON.stringify(text)}`);
  console.log(
    `  error: ${streamError ? String((streamError as Error).message) : 'none'}`,
  );
}

/** Without a retryable model: step 2 fails after the commit. */
await run('retryable call only', createPrimaryModel());

/** With a retryable model: it retries step 2 on the fallback model. */
await run(
  'retryable call + retryable model',
  createRetryableModel({
    model: createPrimaryModel(),
    retries: [httpStatus(529).switch({ model: fallbackModel })],
  }),
);
