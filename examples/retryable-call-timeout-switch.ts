/**
 * Switch models when a call times out, using `createRetryableCall` — the
 * generic retry-loop driver that also backs `createRetryableStream`.
 *
 * The driver owns two things the underlying call cannot do for itself: it picks
 * the model for each attempt, and it mints a FRESH per-attempt deadline
 * (`AbortSignal.timeout`). That freshness is the whole point — attempt 1's clock
 * is already spent by the time it fails, so a re-run has to start from a new
 * signal or it would abort instantly. The call function just wires
 * `attempt.model` + `attempt.abortSignal` into whatever it invokes; here that is
 * `generateText`.
 *
 * When attempt 1's model hangs past the deadline, its signal fires a
 * `TimeoutError`; `timeout().switch({ model, timeout })` matches it and re-runs
 * with the fallback under a fresh deadline.
 *
 * The timeout is simulated (no API key needed): the slow model resolves only
 * when its abort signal fires, so it always trips whatever deadline the driver
 * sets for the attempt.
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
 * A model that answers only after 5s — a stand-in for a slow upstream. The
 * driver's much shorter deadline pre-empts it: when the attempt's `abortSignal`
 * fires (its reason is a `TimeoutError`), `doGenerate` rejects with it. The
 * pending 5s timer also keeps the event loop alive until the deadline fires,
 * which a real network call would do on its own.
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
 * The driver hands each attempt its model plus a fresh `abortSignal`. Forward
 * both into `generateText`; `maxRetries: 0` leaves all retrying to the driver.
 * The first attempt's deadline comes from the run option below.
 */
const result = await run(
  (attempt) =>
    generateText({
      model: attempt.model,
      abortSignal: attempt.abortSignal,
      prompt: 'Invent a new holiday and describe its traditions.',
      maxRetries: 0,
    }),
  { timeout: 100 },
);

console.log(`slow.doGenerate: ${slow.doGenerate.mock.calls.length}`);
console.log(`fast.doGenerate: ${fast.doGenerate.mock.calls.length}`);
console.log(`text: ${JSON.stringify(result.text)}`);
console.log(
  '\nTakeaway: the driver re-runs the whole call with a fresh per-attempt\ndeadline, so a timeout on one model fails over cleanly to the next.',
);
process.exit(0);
