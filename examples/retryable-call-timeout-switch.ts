/**
 * Switch models when a call times out, using `createRetryableCall` — the
 * generic retry-loop driver that also backs `createRetryableStream`.
 *
 * The driver re-runs the whole call for each attempt and hands the call function
 * a FRESH per-attempt deadline as `attempt.timeout` (the run's timeout first,
 * then each fallback's own `.switch({ timeout })`). Freshness is the point:
 * attempt 1's clock is spent by the time it fails, so a re-run needs a new
 * deadline or it would time out instantly. The call function applies that number
 * however its call takes one — `generateText` has a native `timeout`, so it just
 * forwards it. (`attempt.abortSignal` is separate: the caller's own cancellation,
 * passed through untouched.)
 *
 * When attempt 1's model hangs past its deadline, `generateText` throws a
 * `TimeoutError`; `timeout().switch({ model, timeout })` matches it and re-runs
 * with the fallback under its own fresh deadline.
 *
 * The slowness is simulated (no API key needed): the slow model only answers
 * after 5s, so any shorter deadline trips first.
 *
 * Run:
 *   pnpm build && pnpm tsx examples/retryable-call-timeout-switch.ts
 */
import { generateText } from 'ai';
import { createRetryableCall } from 'ai-retry/experimental/call';
import { timeout } from 'ai-retry/language-model';
import { Language, MockLanguageModel } from 'ai-test-kit/language';
import type { LanguageModelV4CallOptions } from '@ai-sdk/provider';

/**
 * A model that answers only after 5s — a stand-in for a slow upstream. A shorter
 * deadline pre-empts it: `generateText`'s `timeout` aborts the call's signal, so
 * `doGenerate` rejects with the reason (a `TimeoutError`). The pending 5s timer
 * also keeps the event loop alive until the deadline fires, which a real network
 * call would do on its own.
 */
const slowModel = () =>
  MockLanguageModel.from({
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
  });

const slow = slowModel();
const fast = MockLanguageModel.from(
  'Leap-Second Day: everyone pauses together for one shared second.',
);

const run = createRetryableCall({
  model: slow,
  retries: [
    /** On a timeout, switch to the fast model under a fresh 1s deadline. */
    timeout().switch({ model: fast, timeout: 1_000 }),
  ],
});

/**
 * The driver hands each attempt its model, a fresh `timeout`, and the caller's
 * `abortSignal` (here none). Forward the timeout to `generateText`'s own timeout
 * option and the signal for cancellation; `maxRetries: 0` leaves retrying to the
 * driver. The first attempt's deadline comes from the run option below.
 */
const result = await run(
  (attempt) =>
    generateText({
      model: attempt.model,
      timeout: attempt.timeout,
      abortSignal: attempt.abortSignal,
      prompt: 'Invent a new holiday and describe its traditions.',
      maxRetries: 0,
    }),
  { timeout: 100 },
);

console.log(`slow.doGenerate: ${slow.doGenerate.mock.calls.length}`);
console.log(`fast.doGenerate: ${fast.doGenerate.mock.calls.length}`);
console.log(`text: ${JSON.stringify(result.text)}`);
process.exit(0);
