import type { LanguageModelResult } from '../../types.js';
import {
  createRetryableCall,
  type RetryableCallOptions,
  type RetryCallAttempt,
  type RetryCallRunOptions,
  type ResultRetry,
} from '../call/create-retryable-call.js';
import { detectStreamCommit } from './detect-stream-commit.js';

/**
 * The minimal shape a stream result must expose: a `fullStream` that can be
 * read to detect when the attempt commits. The stream must be safe to read
 * independently of the caller's own consumption (tee semantics), as the AI
 * SDK's `result.fullStream` getter is — see {@link detectStreamCommit}.
 */
export type StreamResult = { fullStream: ReadableStream<unknown> };

/**
 * Options for {@link createRetryableStream}.
 */
export type RetryableStreamOptions = RetryableCallOptions;

/**
 * Runs a stream-producing function with retry/fail-over, deciding the outcome
 * by reading the result's `fullStream` (no SDK callbacks). Generic over the
 * result type, so it returns whatever `streamFn` returns once an attempt
 * commits.
 */
export type RetryableStream = <RESULT extends StreamResult>(
  streamFn: (attempt: RetryCallAttempt) => RESULT | Promise<RESULT>,
  runOptions?: RetryCallRunOptions,
) => Promise<RESULT>;

/**
 * Make a stream call retryable at the call level, detecting commit/fail-over
 * purely from the result's `fullStream`.
 *
 * For each attempt it invokes `streamFn` (which should build its stream with
 * `attempt.model` and `attempt.abortSignal`), then reads a tee of the result's
 * `fullStream` up to the first content part. If the stream fails *before*
 * content — an error part, or an `abort` part from a `streamText`-level
 * deadline (`timeout.chunkMs`/`stepMs`/`totalMs` or an inbound `abortSignal`) —
 * it re-runs the whole call with the next model, which is the only place such a
 * deadline can fail over (the underlying call has already torn its stream down;
 * see issue #50). Once a content part is seen the attempt is committed and
 * cannot fail over.
 *
 * Result-based retries work too, at parity with the model layer: a `finish`
 * part seen before any content (e.g. a `content-filter` reason with no output)
 * is evaluated against the configured retryables as a result attempt. As below
 * the model layer, only *function* retryables (`contentFilterTriggered(...)`)
 * match a result attempt — a plain fallback model does not — and a no-match
 * returns the (empty) result rather than failing over. Schema-mismatch needs
 * the generated text, which only exists after commit, so it cannot apply here.
 *
 * Decoupled from `streamText`: it depends only on the result exposing a
 * re-readable `fullStream`. Pass a `streamFn` that returns a `streamText` (or
 * `streamObject`) result to make that call retryable at the call level.
 *
 * Returns the winning attempt's result unchanged, so the caller drives the body
 * (`fullStream`, `toUIMessageStreamResponse()`, …) with back-pressure preserved
 * past the commit point.
 */
export function createRetryableStream(
  options: RetryableStreamOptions,
): RetryableStream {
  const run = createRetryableCall(options);

  return <RESULT extends StreamResult>(
    streamFn: (attempt: RetryCallAttempt) => RESULT | Promise<RESULT>,
    runOptions?: RetryCallRunOptions,
  ) =>
    run<RESULT>(async (attempt) => {
      const result = await streamFn(attempt);
      const finishReason = await detectStreamCommit(result.fullStream, attempt);

      /**
       * The stream finished before any content with a finish reason (e.g.
       * `content-filter`). Hand it to the driver as a result-based retry so the
       * configured retryables evaluate it the same way they would below the
       * model layer. A minimal synthetic result is enough — result-based
       * retryables read `finishReason` (and `content`, empty before commit).
       */
      if (finishReason !== undefined) {
        const signal: ResultRetry<RESULT> = {
          type: 'result',
          result: {
            content: [],
            finishReason: { unified: finishReason, raw: undefined },
          } as unknown as LanguageModelResult,
          value: result,
        };
        throw signal;
      }

      return result;
    }, runOptions);
}
