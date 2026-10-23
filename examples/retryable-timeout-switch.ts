/**
 * Switch from a slow model to a fast model on a timeout, on the call layer.
 *
 * The `timeout` argument belongs to the call, not to the model. When it fires
 * in a `streamText` call, the AI SDK stops the call and discards the result of
 * a retry below it, so a retryable model cannot recover it. A retryable call
 * runs the whole call again with the next model, so it recovers the timeout for
 * both `generateText` and `streamText`.
 *
 * `.switch({ timeout })` gives the retry a fresh deadline, because the deadline
 * of the first attempt is used up when it fails.
 *
 * Offline: all models are mocks, no API key needed.
 *
 * Run:
 *   pnpm build && pnpm tsx examples/retryable-timeout-switch.ts
 *
 * Output:
 *   generateText: A fast answer.
 *   streamText: A fast answer.
 */
import type { LanguageModelV4CallOptions } from '@ai-sdk/provider';
import { retryableGenerateText } from 'ai-retry/generate-text';
import { timeout } from 'ai-retry/generate-text/conditions';
import { retryableStreamText } from 'ai-retry/stream-text';
import { Language, MockLanguageModel } from 'ai-test-kit/language';

/**
 * `doGenerate` answers after 5 seconds, and `doStream` opens a stream that sends
 * no content. Both stop when their abort signal fires, so a shorter deadline
 * stops them before they commit.
 */
const slowModel = MockLanguageModel.from(
  {
    doGenerate: {
      content: [Language.text('A slow answer.')],
      delayInMs: 5_000,
    },
    doStream: async ({ abortSignal }: LanguageModelV4CallOptions) => ({
      stream: new ReadableStream({
        start(controller) {
          controller.enqueue(Language.streamStart());
          abortSignal?.addEventListener(
            'abort',
            () => controller.error(abortSignal.reason),
            { once: true },
          );
        },
      }),
    }),
  },
  { provider: 'openai', modelId: 'gpt-4o' },
);

const fastModel = MockLanguageModel.from('A fast answer.', {
  provider: 'openai',
  modelId: 'gpt-4o-mini',
});

/**
 * On a timeout, switch to the fast model with a fresh deadline of 1 second.
 *
 * Both functions can use this list. `timeout()` reads only the error, so it is
 * not tied to the entry point that it comes from. A `result()` or
 * `finishReason()` condition is tied to its entry point.
 */
const retry = [timeout().switch({ model: fastModel, timeout: 1_000 })];

const prompt = 'Invent a new holiday and describe its traditions.';

const generated = await retryableGenerateText({
  model: slowModel,
  prompt,
  timeout: { totalMs: 100 },
  retry,
});

console.log(`generateText: ${generated.text}`);

/** The stream sends no content, so the deadline fires before the commit. */
const streamed = await retryableStreamText({
  model: slowModel,
  prompt,
  timeout: { firstChunkMs: 100 },
  retry,
});

let text = '';
for await (const chunk of streamed.textStream) text += chunk;

console.log(`streamText: ${text}`);
