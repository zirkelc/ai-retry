import {
  createRetryableCall,
  type RetryableCallOptions,
  type RetryCallAttempt,
  type RetryCallRunOptions,
} from '../call/create-retryable-call.js';
import { type CommitGate, detectStreamCommit } from './detect-stream-commit.js';

/**
 * The minimal shape a stream result must expose: a re-readable stream of parts
 * to detect when the attempt commits. `streamText` (AI SDK v7) exposes it as
 * `stream`; `streamObject` (and pre-v7 results) as `fullStream`. Either is
 * accepted, preferring `stream`. The stream must be safe to read independently
 * of the caller's own consumption (tee semantics), as those getters are — see
 * {@link detectStreamCommit}.
 */
export type StreamResult =
  | { stream: ReadableStream<unknown> }
  | { fullStream: ReadableStream<unknown> };

/** Resolve the re-readable part stream, preferring the v7 `stream` getter. */
const resolveStream = (result: StreamResult): ReadableStream<unknown> =>
  'stream' in result ? result.stream : result.fullStream;

/**
 * Options for {@link createRetryableStream}.
 */
export type RetryableStreamOptions = RetryableCallOptions & {
  /**
   * Optional gate that moves the *text*-commit boundary later by buffering
   * leading `text-delta` parts until it can tell a real answer from a canned
   * refusal. Use {@link refusalGate} to fail over from a natural-language
   * refusal (`finishReason: 'stop'`, no error) that the default first-delta
   * commit would otherwise lock in. See {@link CommitGate}.
   */
  commitGate?: CommitGate;
};

/**
 * Runs a stream-producing function with retry/fail-over, deciding the outcome
 * by reading the result's part stream (no SDK callbacks). Generic over the
 * result type, so it returns whatever `streamFn` returns once an attempt
 * commits.
 */
export type RetryableStream = <RESULT extends StreamResult>(
  streamFn: (attempt: RetryCallAttempt) => RESULT | Promise<RESULT>,
  runOptions?: RetryCallRunOptions,
) => Promise<RESULT>;

/**
 * Make a stream call retryable at the call level, detecting commit/fail-over
 * purely from the result's part stream (`stream`, or `fullStream` for
 * `streamObject`).
 *
 * For each attempt it invokes `streamFn` (which should build its stream with
 * `attempt.model` and `attempt.abortSignal`), then reads a tee of the result's
 * part stream up to the first content part. If the stream fails *before*
 * content — an error part, or an `abort` part from a `streamText`-level
 * deadline (`timeout.chunkMs`/`stepMs`/`totalMs` or an inbound `abortSignal`) —
 * it re-runs the whole call with the next model, which is the only place such a
 * deadline can fail over (the underlying call has already torn its stream down;
 * see issue #50). Once a content part is seen the attempt is committed and
 * cannot fail over.
 *
 * Error-based only. Result-based conditions (a `content-filter` finish reason,
 * a schema mismatch) are *not* handled here — they recover best below
 * `streamText`, and they compose: pass a `createRetryable(...)` (with the
 * relevant result-based retryables) as the `model`, and let this wrapper handle
 * errors and deadlines around the call.
 *
 * Decoupled from `streamText`: it depends only on the result exposing a
 * re-readable part stream. Pass a `streamFn` that returns a `streamText` (or
 * `streamObject`) result to make that call retryable at the call level.
 *
 * Returns the winning attempt's result unchanged, so the caller drives the body
 * (`stream`, `toUIMessageStreamResponse()`, …) with back-pressure preserved
 * past the commit point.
 */
export function createRetryableStream(
  options: RetryableStreamOptions,
): RetryableStream {
  const { commitGate, ...callOptions } = options;
  const run = createRetryableCall(callOptions);

  return <RESULT extends StreamResult>(
    streamFn: (attempt: RetryCallAttempt) => RESULT | Promise<RESULT>,
    runOptions?: RetryCallRunOptions,
  ) =>
    run<RESULT>(async (attempt) => {
      const result = await streamFn(attempt);
      await detectStreamCommit(resolveStream(result), attempt, commitGate);
      return result;
    }, runOptions);
}
