import {
  createRetryableCall,
  type RetryableCallOptions,
  type RetryCallAttempt,
  type RetryCallRunOptions,
} from '../call/create-retryable-call.js';
import type {
  LanguageModel,
  RetryAttempt,
  RetryCallOptions,
} from '../../types.js';
import { detectStreamCommit } from './detect-stream-commit.js';

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
 * The attempt that committed: the one whose first content part reached the
 * stream, after which no further fail-over is possible.
 *
 * Language-model-shaped rather than generic, since commit is detected from the
 * AI SDK stream protocol and has no meaning for the other model kinds.
 */
export type RetryStreamCommitAttempt = {
  /** The model whose attempt committed. */
  model: LanguageModel;
  /** The per-attempt overrides applied to the committed attempt. */
  options: RetryCallOptions<LanguageModel>;
};

/**
 * The context passed to `onCommit`, with the committed attempt and the
 * attempts that were retried before it.
 */
export type RetryStreamCommitContext = {
  /** The attempt that committed. */
  current: RetryStreamCommitAttempt;
  /**
   * The preceding attempts that were retried, in order. Empty when the first
   * attempt committed. The committed attempt is `current` and is not repeated
   * here.
   */
  attempts: Array<RetryAttempt<LanguageModel>>;
};

/**
 * Options for {@link createRetryableStream}.
 *
 * The call driver's `onComplete` is replaced by `onCommit`, which is the same
 * callback observed at a later boundary: this wrapper's attempt does not return
 * until the stream has committed, so "the call function returned" and "the
 * first content part arrived" are the same moment here.
 */
export type RetryableStreamOptions = Omit<
  RetryableCallOptions,
  'onComplete'
> & {
  /**
   * Called once an attempt commits — its first content part reached the
   * stream — so the wrapper will not fail over again. Not when the stream
   * finishes: past this point the stream is the caller's, and an error during
   * consumption fires nothing. For a hook that waits for the stream to
   * actually finish, use the model wrapper's `onSuccess` below this one, or
   * the SDK's own `onFinish`.
   */
  onCommit?: (context: RetryStreamCommitContext) => void;
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
 *
 * The outcome hooks follow that same boundary: `onCommit` fires once an attempt
 * commits — the first content part, not the end of the stream — and `onFailure`
 * fires when every attempt failed before committing. Past the commit point the
 * caller owns the stream, so an error during consumption fires neither.
 */
export function createRetryableStream(
  options: RetryableStreamOptions,
): RetryableStream {
  const { onCommit, ...callOptions } = options;
  /**
   * The attempt only returns once the stream has committed, so the driver's
   * completion callback is exactly this wrapper's commit callback.
   */
  const run = createRetryableCall({ ...callOptions, onComplete: onCommit });

  return <RESULT extends StreamResult>(
    streamFn: (attempt: RetryCallAttempt) => RESULT | Promise<RESULT>,
    runOptions?: RetryCallRunOptions,
  ) =>
    run<RESULT>(async (attempt) => {
      const result = await streamFn(attempt);
      await detectStreamCommit(resolveStream(result), attempt);
      return result;
    }, runOptions);
}
