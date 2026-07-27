/**
 * Fail over from a slow model to a fast one on a timeout, at the CALL level, for
 * both `generateText` (via `createRetryableCall`) and `streamText` (via
 * `createRetryableStream`).
 *
 * A `generateText` / `streamText` timeout lives ON the call, above the model, so
 * `createRetryableModel` (which retries below) can't recover it: once the
 * deadline fires the SDK tears the call down and discards whatever a lower retry
 * produced (issue #50). The call-level drivers re-run the whole call with the
 * next model, so fail-over works.
 *
 * Each driver hands the call function a fresh per-attempt `timeout` (the run's
 * timeout first, then the fallback's own `.switch({ timeout })`), the attempt's
 * `model`, and the caller's `abortSignal`. Apply the timeout however the call
 * takes one: `generateText` has a native `timeout`; for `streamText` use a
 * pre-content deadline (`firstChunkMs`). `maxRetries: 0` leaves retrying to the
 * driver.
 *
 * Offline (no API key): the slow model only answers after 5s (generate) or
 * stalls its stream open with no content (stream), so any shorter deadline trips
 * first and fails over to the fast model.
 *
 * Run:
 *   pnpm build && pnpm tsx examples/retryable-timeout-switch.ts
 */
import { generateText, streamText } from 'ai';
import { createRetryableCall } from 'ai-retry/experimental/call';
import { createRetryableStream } from 'ai-retry/experimental/stream';
import { timeout } from 'ai-retry/language-model';
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
const retries = [timeout().switch({ model: fastModel, timeout: 1_000 })];

/**
 * generateText: `createRetryableCall` forwards `attempt.timeout` to
 * generateText's own `timeout` option.
 */
const runGenerate = createRetryableCall({ model: slowModel, retries });

const generated = await runGenerate(
  (attempt) =>
    generateText({
      model: attempt.model,
      timeout: attempt.timeout,
      abortSignal: attempt.abortSignal,
      prompt: 'Invent a new holiday and describe its traditions.',
      maxRetries: 0,
    }),
  /** First-attempt deadline; the fallback uses its own `.switch({ timeout })`. */
  { timeout: 100 },
);

console.log(`generateText -> ${JSON.stringify(generated.text)}`);

/**
 * streamText: `createRetryableStream` applies `attempt.timeout` to the
 * time-to-first-chunk window; the slow stream stalls before content, so the
 * deadline trips pre-commit and fails over.
 */
const runStream = createRetryableStream({ model: slowModel, retries });

const streamed = await runStream(
  (attempt) =>
    streamText({
      model: attempt.model,
      timeout: { firstChunkMs: attempt.timeout },
      abortSignal: attempt.abortSignal,
      prompt: 'Invent a new holiday and describe its traditions.',
      maxRetries: 0,
    }),
  { timeout: 100 },
);

let text = '';
for await (const chunk of streamed.textStream) text += chunk;
console.log(`streamText   -> ${JSON.stringify(text)}`);

process.exit(0);
