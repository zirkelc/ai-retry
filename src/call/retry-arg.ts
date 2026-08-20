import type {
  AnyModel,
  OnRetryOverrides,
  RetryTelemetrySettings,
  RetryTimeout,
} from '../types.js';
import type {
  CallRetries,
  CallRetryAttempt,
  CallRetryContext,
} from './types.js';

/**
 * The attempt that ended the retry loop.
 *
 * A third way for an attempt to end, alongside the two a retryable sees: it
 * errored, it produced a result that was judged and retried, or — this — it
 * settled the operation.
 *
 * For a streaming entry point that means the attempt *committed*: its first
 * content part reached the stream. Whether the stream then ends well is past
 * this library's reach, and past anything it could have retried, so it is not
 * reported here. Use `streamText`'s own `onFinish` for that.
 */
export type CallSuccessfulAttempt<MODEL extends AnyModel, RESULT> = {
  type: 'success';
  /** The model that settled the operation. */
  model: MODEL;
  /** The entry point's own result, exactly as the caller receives it. */
  result: RESULT;
};

/** An attempt, however it ended. */
export type CallSettledAttempt<MODEL extends AnyModel, RESULT, COMMIT> =
  | CallRetryAttempt<MODEL, COMMIT>
  | CallSuccessfulAttempt<MODEL, RESULT>;

/**
 * What one call amounted to, reported once, whether it succeeded or not.
 *
 * The only terminal hook, deliberately: separate success and failure callbacks
 * make the common question — how often does retrying actually rescue a call —
 * a matter of correlating two handlers, and of knowing which of them counts
 * the attempt that ended things. This carries both facts in one place.
 *
 * `attempts` always holds every attempt, the terminal one included, so its
 * length is the attempt count rather than a retry count that means something
 * different on each path:
 *
 * ```ts
 * onSettled: ({ outcome, attempts }) =>
 *   metrics.increment(
 *     `ai_retry.${outcome}.${attempts.length > 1 ? 'retried' : 'first_try'}`,
 *   );
 * ```
 *
 * Mirrors the operation span field for field — `ai_retry.outcome`,
 * `ai_retry.attempts`, `ai_retry.model.final` — so a metric built on this and
 * one built on telemetry agree by construction.
 */
export type CallSettledEvent<
  MODEL extends AnyModel,
  RESULT,
  COMMIT = RESULT,
> = {
  /** Whether the call produced a result or threw. */
  outcome: 'success' | 'failure';
  /** The model that settled it, or the one whose failure ended the loop. */
  model: MODEL;
  /**
   * Every attempt, in order, ending with the one that settled the operation.
   * Never empty: a call always makes at least one attempt.
   */
  attempts: Array<CallSettledAttempt<MODEL, RESULT, COMMIT>>;
  /** The result, on success. */
  result?: RESULT;
  /**
   * What the call rejects with, on failure: a `RetryError` wrapping every
   * attempt's error when more than one was made, otherwise the original.
   */
  error?: unknown;
};

/**
 * Retry configuration in its full form.
 *
 * Three shapes are threaded through:
 *
 * - `INPUT` is inferred from the `retries` array, and constrained by the entry
 *   point to its own argument shape. Inferring it is what catches an override
 *   built for a *different* entry point — a `CallRetryable` carrying
 *   `options: { values }` cannot satisfy a bound of `generateText` arguments.
 * - `OVERRIDE` is that bound, named directly. `onRetry` is typed against it
 *   rather than against `INPUT`, so its return value neither competes with the
 *   `retries` array to define `INPUT` nor has to repeat every field some
 *   listed retry happens to set.
 * - `RESULT` is what the entry point returns, which `onSettled` reports.
 * - `COMMIT` is what a result condition judges. It is the result for most entry
 *   points, and defaults to it; see `CallRetryContext` for when it is not.
 */
export type CallRetryOptions<
  MODEL extends AnyModel,
  INPUT,
  OVERRIDE,
  RESULT,
  COMMIT = RESULT,
  TIMEOUT extends RetryTimeout = number,
> = {
  /** Retry handlers and fallback models, evaluated on each failed attempt. */
  retries: CallRetries<MODEL, INPUT, COMMIT, TIMEOUT>;
  /**
   * Bypass the retry machinery entirely, making the call behave exactly as a
   * direct call to the underlying entry point — including the SDK's own
   * `maxRetries` default, which is otherwise disabled (see `runRetryLoop`).
   */
  disabled?: boolean | (() => boolean);
  /**
   * Experimental. Can change in patch versions without warning.
   *
   * Telemetry configuration. When enabled, emits OpenTelemetry spans for the
   * operation and each attempt. Requires `@opentelemetry/api`.
   */
  telemetry?: RetryTelemetrySettings;
  /** Called for every failed attempt, whether or not a retry follows. */
  onError?: (context: CallRetryContext<MODEL, COMMIT>) => void;
  /**
   * Called after a retry has been decided and the next model selected, but
   * before the retry call is issued. May return overrides for the upcoming
   * attempt.
   *
   * Per-field precedence for the upcoming call, highest first:
   * `onRetry` return value → `Retry.options` → the call's own arguments.
   */
  onRetry?: (
    context: CallRetryContext<MODEL, COMMIT>,
  ) =>
    | void
    | OnRetryOverrides<MODEL, OVERRIDE>
    | Promise<void | OnRetryOverrides<MODEL, OVERRIDE>>;
  /**
   * Called once the call terminally fails: no retry matched, every candidate
   * was tried, the caller's signal was already aborted, or the caller aborted
   * during a backoff delay.
   *
   * Reports attempt failures only, so it stays silent for a rejection no
   * attempt caused — a callback of your own throwing, for instance. Also
   * silent when retries are disabled.
   */
  /** Called once, with what the whole call amounted to. */
  onSettled?: (event: CallSettledEvent<MODEL, RESULT, COMMIT>) => void;
};

/**
 * The `retry` argument.
 *
 * The bare array is the common form; the object form adds hooks, telemetry and
 * the disable switch. Grouping everything under one key keeps exactly one name
 * in collision range should the SDK add arguments of its own.
 *
 * @example
 * retry: [serviceOverloaded(fallback)]
 * retry: { retries: [fallback], onRetry: (ctx) => log(ctx) }
 */
export type CallRetryArg<
  MODEL extends AnyModel,
  INPUT,
  OVERRIDE,
  RESULT,
  COMMIT = RESULT,
  TIMEOUT extends RetryTimeout = number,
> =
  | CallRetries<MODEL, INPUT, COMMIT, TIMEOUT>
  | CallRetryOptions<MODEL, INPUT, OVERRIDE, RESULT, COMMIT, TIMEOUT>;

/**
 * Normalize either `retry` form (or its absence) into the full options object.
 */
export function toCallRetryOptions<
  MODEL extends AnyModel,
  INPUT,
  OVERRIDE,
  RESULT,
  COMMIT = RESULT,
  TIMEOUT extends RetryTimeout = number,
>(
  retry:
    | CallRetryArg<MODEL, INPUT, OVERRIDE, RESULT, COMMIT, TIMEOUT>
    | undefined,
): CallRetryOptions<MODEL, INPUT, OVERRIDE, RESULT, COMMIT, TIMEOUT> {
  if (retry === undefined) return { retries: [] };
  return Array.isArray(retry) ? { retries: retry } : retry;
}
