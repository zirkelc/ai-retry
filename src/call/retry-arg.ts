import type {
  AnyModel,
  OnRetryOverrides,
  RetryTelemetrySettings,
} from '../types.js';
import type {
  CallFailureContext,
  CallRetries,
  CallRetryAttempt,
  CallRetryContext,
} from './types.js';
import type { CallFinishReason } from './types.js';

/**
 * The attempt that produced the returned result.
 *
 * For a streaming entry point this is the attempt that *committed* — its first
 * content part reached the stream — not one that finished well. Past that
 * point the stream belongs to the caller, and an error while consuming it
 * fires nothing here; use the SDK's own `onFinish` for that.
 */
export type CallSuccessAttempt<MODEL extends AnyModel, RESULT> = {
  type: 'success';
  /** The model that produced the result. */
  model: MODEL;
  /** The entry point's own result, exactly as the caller receives it. */
  result: RESULT;
  /**
   * The unified finish reason, when it was known before the result was
   * handed over. Absent for a committed stream, whose finish reason is only
   * decided during consumption.
   */
  finishReason?: CallFinishReason;
};

/**
 * The context passed to `onSuccess`, with the attempt that produced the result
 * and the attempts retried before it.
 */
export type CallSuccessContext<MODEL extends AnyModel, RESULT> = {
  /** The attempt that produced the result. */
  current: CallSuccessAttempt<MODEL, RESULT>;
  /**
   * The preceding attempts that were retried, in order. Empty when the first
   * attempt succeeded. The successful attempt is `current` and is not repeated
   * here.
   */
  attempts: Array<CallRetryAttempt<MODEL>>;
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
 * - `RESULT` is what the entry point returns, which `onSuccess` receives.
 */
export type CallRetryOptions<
  MODEL extends AnyModel,
  INPUT,
  OVERRIDE,
  RESULT,
> = {
  /** Retry handlers and fallback models, evaluated on each failed attempt. */
  retries: CallRetries<MODEL, INPUT>;
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
  onError?: (context: CallRetryContext<MODEL>) => void;
  /**
   * Called after a retry has been decided and the next model selected, but
   * before the retry call is issued. May return overrides for the upcoming
   * attempt.
   *
   * Per-field precedence for the upcoming call, highest first:
   * `onRetry` return value → `Retry.options` → the call's own arguments.
   */
  onRetry?: (
    context: CallRetryContext<MODEL>,
  ) =>
    | void
    | OnRetryOverrides<MODEL, OVERRIDE>
    | Promise<void | OnRetryOverrides<MODEL, OVERRIDE>>;
  /** Called once an attempt produces the result the caller receives. */
  onSuccess?: (context: CallSuccessContext<MODEL, RESULT>) => void;
  /**
   * Called once the call terminally fails: no retry matched, every candidate
   * was tried, the caller's signal was already aborted, or the caller aborted
   * during a backoff delay.
   *
   * Reports attempt failures only, so it stays silent for a rejection no
   * attempt caused — a callback of your own throwing, for instance. Also
   * silent when retries are disabled.
   */
  onFailure?: (context: CallFailureContext<MODEL>) => void;
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
export type CallRetryArg<MODEL extends AnyModel, INPUT, OVERRIDE, RESULT> =
  | CallRetries<MODEL, INPUT>
  | CallRetryOptions<MODEL, INPUT, OVERRIDE, RESULT>;

/**
 * Normalize either `retry` form (or its absence) into the full options object.
 */
export function toCallRetryOptions<
  MODEL extends AnyModel,
  INPUT,
  OVERRIDE,
  RESULT,
>(
  retry: CallRetryArg<MODEL, INPUT, OVERRIDE, RESULT> | undefined,
): CallRetryOptions<MODEL, INPUT, OVERRIDE, RESULT> {
  if (retry === undefined) return { retries: [] };
  return Array.isArray(retry) ? { retries: retry } : retry;
}
