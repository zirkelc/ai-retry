/**
 * Fail over from a slow model to a fast one on a timeout, at the CALL level,
 * for both `retryableGenerateText` and `retryableStreamText`.
 *
 * A `generateText` / `streamText` timeout lives ON the call, above the model, so
 * `createRetryableModel` (which retries below) can't recover it: once the
 * deadline fires the SDK tears the call down and discards whatever a lower retry
 * produced (issue #50). The call-level functions re-run the whole call with the
 * next model, so fail-over works.
 *
 * Each takes the entry point's own arguments plus `retry`. The deadline goes
 * where that entry point takes one — `timeout` for these two — and a matched
 * `.switch({ timeout })` gives the retry a fresh one, since the first attempt's
 * clock is already spent by the time it fails.
 *
 * Offline (no API key): the slow model only answers after 5s (generate) or
 * stalls its stream open with no content (stream), so any shorter deadline trips
 * first and fails over to the fast model.
 *
 * Run:
 *   pnpm build && pnpm tsx examples/retryable-timeout-switch.ts
 */
import { retryableGenerateText, retryableStreamText } from 'ai-retry';
import { timeout } from 'ai-retry/call/language-model/conditions';
import { Language, MockLanguageModel } from 'ai-test-kit/language';
import type { LanguageModelV4CallOptions } from '@ai-sdk/provider';

/**
 * A slow upstream stand-in: `doGenerate` answers only after 5s, and `doStream`
 * opens its stream then never sends content. Both reject when their abort signal
 * fires, which is what the per-attempt deadline does, so any shorter deadline
 * pre-empts either call before it can commit.
 */
const slowModel = MockLanguageModel.from({
  doGenerate: ({ abortSignal }: LanguageModelV4CallOptions) =>
    new Promise((resolve, reject) => {
      const timer = setTimeout(
        () => resolve(Language.result('...eventually, a slow answer.')),
        5_000,
      );
      abortSignal?.addEventListener(
        'abort',
        () => {
          clearTimeout(timer);
          reject(abortSignal.reason);
        },
        { once: true },
      );
    }),
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
});

const fastModel = MockLanguageModel.from('A fast, complete answer.');

/** On a timeout, switch to the fast model under a fresh 1s deadline. */
const retry = [timeout().switch({ model: fastModel, timeout: 1_000 })];

const prompt = 'Invent a new holiday and describe its traditions.';

/** generateText: the deadline is the call's own `timeout` argument. */
const generated = await retryableGenerateText({
  model: slowModel,
  prompt,
  timeout: { totalMs: 100 },
  retry,
});

console.log(`generateText -> ${JSON.stringify(generated.text)}`);

/**
 * streamText: a pre-content deadline. The slow stream stalls before emitting
 * anything, so it trips before the attempt can commit and fail-over is still
 * possible.
 */
const streamed = await retryableStreamText({
  model: slowModel,
  prompt,
  timeout: { firstChunkMs: 100 },
  retry,
});

let text = '';
for await (const chunk of streamed.textStream) text += chunk;
console.log(`streamText   -> ${JSON.stringify(text)}`);

process.exit(0);
