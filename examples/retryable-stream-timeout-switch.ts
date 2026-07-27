import { anthropic } from '@ai-sdk/anthropic';
import { openai } from '@ai-sdk/openai';
import { streamText } from 'ai';
import { createRetryableStream } from 'ai-retry/experimental/stream';
import { timeout } from 'ai-retry/language-model';

/**
 * Fail over to another model when a *streamed* call times out, using the
 * experimental `createRetryableStream`.
 *
 * A `streamText` timeout (`timeout.firstChunkMs` / `stepMs` / `totalMs`) or an
 * inbound `abortSignal` cannot be recovered by `createRetryableModel`: that
 * retries below `streamText`, but the timeout lives on the call, and once it
 * fires `streamText` tears the stream down and discards whatever a lower retry
 * produced (issue #50). `createRetryableStream` re-runs the whole call with the
 * next model, so fail-over works.
 *
 * Retry is only possible before the first content part. A timeout that fires
 * before any text has streamed (a stalled connection) fails over; once content
 * has been emitted the attempt is committed and the partial stream is surfaced
 * to the caller instead.
 */
const run = createRetryableStream({
  model: openai('gpt-4o'),
  retries: [
    /** On a pre-content timeout, fail over to Claude under a fresh deadline. */
    timeout().switch({
      model: anthropic('claude-sonnet-4-0'),
      timeout: 30_000,
    }),
  ],
});

/**
 * Each attempt builds its own `streamText` with the attempt's model and signal.
 * The wrapper hands a fresh per-attempt timeout as `attempt.timeout` (the run's
 * timeout first, then the fallback's own `.switch({ timeout })`); apply it to
 * the time-to-first-chunk window. `maxRetries: 0` leaves retrying to the
 * wrapper, and `run` resolves once an attempt commits, so `result` is the
 * winning stream — drive it as usual.
 */
const result = await run(
  (attempt) =>
    streamText({
      model: attempt.model,
      abortSignal: attempt.abortSignal,
      prompt: 'Write a story about a robot...',
      timeout: { firstChunkMs: attempt.timeout },
      maxRetries: 0,
    }),
  /** First-attempt deadline; the fallback uses its own `.switch({ timeout })`. */
  { timeout: 5_000 },
);

for await (const chunk of result.textStream) {
  console.log(chunk);
}
