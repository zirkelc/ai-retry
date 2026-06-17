/**
 * ai-retry telemetry → Langfuse example.
 *
 * Run with: pnpm tsx examples/telemetry/index.ts
 *
 * A flaky primary model reports itself as overloaded (HTTP 529). The retry
 * layer falls back to a second model, and both attempts are exported to
 * Langfuse as OpenTelemetry spans. The example runs once with `generateText`
 * and once with `streamText`; with telemetry enabled on the AI SDK calls too,
 * the Langfuse traces look like:
 *
 *   ai.generateText                          ai.streamText
 *   └─ ai.generateText.doGenerate            └─ ai.streamText.doStream
 *      └─ ai_retry.doGenerate                   └─ ai_retry.doStream
 *         ├─ ai_retry.attempt #1 (529)            ├─ ai_retry.attempt #1 (529)
 *         └─ ai_retry.attempt #2 (ok)             └─ ai_retry.attempt #2 (ok)
 *
 * Requires LANGFUSE_PUBLIC_KEY / LANGFUSE_SECRET_KEY / LANGFUSE_BASE_URL in
 * .env. No LLM provider key is needed — both models are mocks.
 */
import { langfuseSpanProcessor } from './instrumentation.js';
import {
  APICallError,
  generateText,
  simulateReadableStream,
  streamText,
} from 'ai';
import { MockLanguageModelV4 } from 'ai/test';
import {
  createRetryableModel,
  httpStatus,
} from '../../src/language-model/index.js';

const usage = {
  inputTokens: { total: 8, noCache: 8, cacheRead: 0, cacheWrite: 0 },
  outputTokens: { total: 5, text: 5, reasoning: 0 },
};

const overloaded = () =>
  new APICallError({
    message: 'Service overloaded',
    url: 'https://example.com/v1/chat',
    requestBodyValues: {},
    statusCode: 529,
    isRetryable: true,
  });

/** A primary model that always reports itself as overloaded (HTTP 529). */
const flakyModel = new MockLanguageModelV4({
  provider: 'mock',
  modelId: 'flaky-primary',
  doGenerate: async () => {
    throw overloaded();
  },
  doStream: async () => ({
    stream: simulateReadableStream({
      chunks: [
        { type: 'stream-start', warnings: [] },
        { type: 'error', error: overloaded() },
      ],
    }),
  }),
});

/** A reliable fallback model that returns a static answer. */
const reliableModel = new MockLanguageModelV4({
  provider: 'mock',
  modelId: 'reliable-fallback',
  doGenerate: async () => ({
    finishReason: { unified: 'stop', raw: 'stop' },
    usage,
    content: [{ type: 'text', text: 'Hello from the fallback model!' }],
    warnings: [],
  }),
  doStream: async () => ({
    stream: simulateReadableStream({
      chunks: [
        { type: 'stream-start', warnings: [] },
        { type: 'text-start', id: '1' },
        {
          type: 'text-delta',
          id: '1',
          delta: 'Hello from the fallback model!',
        },
        { type: 'text-end', id: '1' },
        {
          type: 'finish',
          finishReason: { unified: 'stop', raw: 'stop' },
          usage,
        },
      ],
    }),
  }),
});

const model = createRetryableModel({
  model: flakyModel,
  retries: [httpStatus(529, 'overloaded').switch({ model: reliableModel })],
  experimental_telemetry: { isEnabled: true, functionId: 'telemetry-example' },
});

/** Disable the AI SDK's own retries so each trace shows one model call. */
const telemetry = { isEnabled: true, functionId: 'telemetry-example' };

const { text } = await generateText({
  model,
  prompt: 'Say hello.',
  maxRetries: 0,
  experimental_telemetry: telemetry,
});
console.log('generateText:', text);

const result = streamText({
  model,
  prompt: 'Say hello, streaming.',
  maxRetries: 0,
  experimental_telemetry: telemetry,
});
let streamed = '';
for await (const chunk of result.textStream) streamed += chunk;
console.log('streamText:', streamed);

/** Flush spans before the process exits. */
await langfuseSpanProcessor.forceFlush();
console.log('Spans flushed to Langfuse.');
