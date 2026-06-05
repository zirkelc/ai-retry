import {
  createRetryableCall,
  type RetryableCallOptions,
  type RetryCallAttempt,
  type RetryCallRunOptions,
} from '../call/create-retryable-call.js';
import {
  type ClassifyStreamPart,
  classifyStreamTextPart,
  detectStreamCommit,
} from './detect-stream-commit.js';

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
export interface RetryableStreamOptions extends RetryableCallOptions {
  /**
   * Classifies each `fullStream` part to decide commit vs. fail-over.
   *
   * @default classifyStreamTextPart — the AI SDK `fullStream` protocol.
   */
  classifyPart?: ClassifyStreamPart;
}

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
 * Decoupled from `streamText`: it depends only on the result exposing a
 * re-readable `fullStream`. Use {@link createRetryableStreamText} for a typed
 * `streamText` drop-in.
 *
 * Returns the winning attempt's result unchanged, so the caller drives the body
 * (`fullStream`, `toUIMessageStreamResponse()`, …) with back-pressure preserved
 * past the commit point.
 */
export function createRetryableStream(
  options: RetryableStreamOptions,
): RetryableStream {
  const { classifyPart = classifyStreamTextPart, ...callOptions } = options;
  const run = createRetryableCall(callOptions);

  return <RESULT extends StreamResult>(
    streamFn: (attempt: RetryCallAttempt) => RESULT | Promise<RESULT>,
    runOptions?: RetryCallRunOptions,
  ) =>
    run<RESULT>(async (attempt) => {
      const result = await streamFn(attempt);
      await detectStreamCommit(result.fullStream, attempt, classifyPart);
      return result;
    }, runOptions);
}
