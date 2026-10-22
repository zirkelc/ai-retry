/**
 * Export the retry attempts to Langfuse as OpenTelemetry spans.
 *
 * The base model fails with HTTP 529. The retryable model retries it, then
 * switches to the fallback model, and each attempt is a span. The example runs
 * one time with `generateText` and one time with `streamText`. The AI SDK spans
 * come from `@ai-sdk/otel`, which `./instrumentation.ts` registers. In Langfuse,
 * the traces look like this:
 *
 *   ai.generateText                          ai.streamText
 *   └─ ai.generateText.doGenerate            └─ ai.streamText.doStream
 *      └─ ai_retry.doGenerate                   └─ ai_retry.doStream
 *         ├─ ai_retry.attempt #1 (529)             ├─ ai_retry.attempt #1 (529)
 *         ├─ ai_retry.attempt #2 (529)             ├─ ai_retry.attempt #2 (529)
 *         └─ ai_retry.attempt #3 (ok)              └─ ai_retry.attempt #3 (ok)
 *
 * Requires: LANGFUSE_PUBLIC_KEY, LANGFUSE_SECRET_KEY and LANGFUSE_BASE_URL in
 * `.env`. All models are mocks, so no LLM provider key is needed.
 *
 * Run:
 *   pnpm build && pnpm tsx examples/telemetry/index.ts
 */
import { langfuseSpanProcessor } from './instrumentation.js';
import { APICallError, generateText, streamText } from 'ai';
import { createRetryableModel, httpStatus } from 'ai-retry/language-model';
import { MockLanguageModel } from 'ai-test-kit/language';

/** Fails with HTTP 529 on every call. */
const baseModel = MockLanguageModel.from(
  new APICallError({
    message: 'Overloaded',
    statusCode: 529,
    url: 'https://api.openai.com/v1/responses',
    requestBodyValues: {},
    isRetryable: true,
  }),
  { provider: 'openai', modelId: 'gpt-5' },
);

const fallbackModel = MockLanguageModel.from('Hello!', {
  provider: 'anthropic',
  modelId: 'claude-opus-4-5',
});

const retryableModel = createRetryableModel({
  model: baseModel,
  retries: [
    httpStatus(529, 'overloaded').retry(),
    httpStatus(529, 'overloaded').switch({ model: fallbackModel }),
  ],
  telemetry: { isEnabled: true },
});

const telemetry = { isEnabled: true, functionId: 'telemetry-example' };

const generated = await generateText({
  model: retryableModel,
  prompt: 'Say hello.',
  maxRetries: 0,
  telemetry,
});

console.log(`generateText: ${generated.text}`);

const streamed = streamText({
  model: retryableModel,
  prompt: 'Say hello, streaming.',
  maxRetries: 0,
  telemetry,
});

let text = '';
for await (const chunk of streamed.textStream) text += chunk;

console.log(`streamText: ${text}`);

/** Send the spans before the process exits. */
await langfuseSpanProcessor.forceFlush();
console.log('Spans sent to Langfuse.');
