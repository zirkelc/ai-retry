/**
 * Switch to a faster model when a `generateText` call times out, on the model layer.
 *
 * The `timeout` argument of `generateText` aborts the request. `timeout()`
 * matches the `TimeoutError`, and `.switch({ timeout })` gives the fallback a
 * fresh deadline. This works for `generateText` only. For `streamText`, use a
 * retryable call, see `examples/retryable-timeout-switch.ts`.
 *
 * Offline: all models are mocks, no API key needed.
 *
 * Run:
 *   pnpm build && pnpm tsx examples/request-timeout.ts
 *
 * Output:
 *   retry with openai/gpt-4o-mini
 *   text: A fast answer.
 */
import { generateText } from 'ai';
import { createRetryableModel, timeout } from 'ai-retry/language-model';
import { Language, MockLanguageModel } from 'ai-test-kit/language';

/** Answers after 5 seconds, so a shorter deadline fires first. */
const slowModel = MockLanguageModel.from(
  { content: [Language.text('A slow answer.')], delayInMs: 5_000 },
  { provider: 'openai', modelId: 'gpt-4o' },
);

const fastModel = MockLanguageModel.from('A fast answer.', {
  provider: 'openai',
  modelId: 'gpt-4o-mini',
});

const retryableModel = createRetryableModel({
  model: slowModel,
  retries: [
    // On a timeout, switch to the fast model with a fresh deadline of 30 seconds
    timeout().switch({ model: fastModel, timeout: 30_000 }),
  ],
  onRetry: ({ current }) => {
    console.log(
      `retry with ${current.model.provider}/${current.model.modelId}`,
    );
  },
});

const result = await generateText({
  model: retryableModel,
  prompt: 'Invent a new holiday and describe its traditions.',
  timeout: 100,
  maxRetries: 0,
});

console.log(`text: ${result.text}`);
