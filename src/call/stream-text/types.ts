import type { ProviderMetadata } from 'ai';
import type {
  CallFinishReason,
  CallLanguageModelUsage,
  CallRetries,
} from '../types.js';
import type { CallRetryOptionsBase, CallSuccessContext } from '../retry-arg.js';
import type { AnyModel, RetryTimeout } from '../../types.js';

/**
 * What a `retryableStreamText` result condition judges: a stream that finished
 * without ever emitting content.
 *
 * Deliberately not the SDK's own `StreamTextResult`, which is what the caller
 * receives. Every field on that is a promise that settles only once the stream
 * has been consumed, and consuming it is precisely what a pre-commit judgement
 * must not do. What is here is read off the stream's terminal parts instead.
 *
 * There is no content, and that is a fact rather than an omission — any content
 * part would have committed the attempt and put it beyond retry. Result
 * conditions here are therefore effectively finish-reason-shaped, a ceiling
 * inherent to streaming rather than to this implementation, and this type says
 * so by declaring no content at all.
 */
export type StreamTextCommitResult = {
  finishReason: CallFinishReason;
  usage: CallLanguageModelUsage;
  providerMetadata: ProviderMetadata | undefined;
};

/**
 * Retry configuration for `retryableStreamText`.
 *
 * Identical to the other entry points' except for the terminal hook, which is
 * `onCommit` rather than `onSuccess`. The rename is not cosmetic: the loop can
 * only report the moment the attempt stopped being recoverable, and for a
 * stream that is the first content part, not a good ending. A stream that
 * commits and then fails in the consumer's hands would have been reported as a
 * success.
 *
 * There is deliberately no end-of-stream hook here. `streamText` already gives
 * the caller `onFinish` and `onError` on the same call, and a stream can carry
 * its own failure as an `error` or `abort` part while completing perfectly
 * normally — so anything this library added would either duplicate those or
 * quietly disagree with them.
 */
export type StreamTextRetryOptions<
  MODEL extends AnyModel,
  INPUT,
  OVERRIDE,
  RESULT,
  TIMEOUT extends RetryTimeout = number,
> = CallRetryOptionsBase<
  MODEL,
  INPUT,
  OVERRIDE,
  StreamTextCommitResult,
  TIMEOUT
> & {
  /**
   * Called once an attempt commits: its first content part has reached the
   * stream, so the stream is the caller's and no further fail-over is possible.
   *
   * Reports that bytes have started, not that the stream ended well. For that,
   * use `streamText`'s own `onFinish`.
   */
  onCommit?: (
    context: CallSuccessContext<MODEL, RESULT, StreamTextCommitResult>,
  ) => void;
};

/**
 * The `retry` argument for `retryableStreamText`: the bare array, or the object
 * form above.
 */
export type StreamTextRetryArg<
  MODEL extends AnyModel,
  INPUT,
  OVERRIDE,
  RESULT,
  TIMEOUT extends RetryTimeout = number,
> =
  | CallRetries<MODEL, INPUT, StreamTextCommitResult, TIMEOUT>
  | StreamTextRetryOptions<MODEL, INPUT, OVERRIDE, RESULT, TIMEOUT>;
