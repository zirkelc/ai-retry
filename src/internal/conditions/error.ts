import { APICallError } from 'ai';
import type { AnyResolvableModel, ModelRetryAttempt } from '../../types.js';
import { isAbortError, isErrorAttempt, isTimeoutError } from '../guards.js';
import { Condition, type LayerContext, type RetryLayer } from './condition.js';
import { or } from './or.js';

/**
 * A pattern accepted by `httpStatus`. Numbers match the response status
 * code; strings match the error message as a substring; regular
 * expressions match against both the stringified status code and the
 * error message.
 */
export type StatusPattern = number | string | RegExp;

/**
 * An error class accepted by `error.isInstance`. Any `Error` subclass
 * constructor works. AI SDK error classes additionally expose a static
 * `isInstance` marker check, which is preferred over `instanceof` when
 * present so matching survives across realms and duplicate installs.
 */
export type ErrorClass = (new (...args: Array<any>) => Error) & {
  isInstance?: (err: unknown) => boolean;
};

/**
 * Build the error-side condition helpers (`error`, `httpStatus`,
 * `timeout`, `aborted`) bound to a specific model family. Consumed by
 * `language-model.ts` and `image-model.ts` so each entry point exposes
 * helpers whose `MODEL` generic is constrained to the right family.
 */
export function createErrorAPI<
  BOUND extends AnyResolvableModel,
  LAYER extends RetryLayer = 'model',
>() {
  /**
   * Build a condition from a predicate over the current error. The
   * predicate runs only when the current attempt failed with an error;
   * result attempts return false.
   *
   * **Important:** returns a `Condition`, not a retryable. Call
   * `.switch()` or `.retry()` to plug it into `retries: [...]`.
   *
   * @example
   * error<MODEL, APICallError>(
   *   (e) => APICallError.isInstance(e) && e.statusCode === 418,
   * ).switch({ model: fallback })
   */
  function error<MODEL extends BOUND = BOUND, E = unknown>(
    predicate: (
      err: E,
      ctx: LayerContext<LAYER, MODEL>,
    ) => boolean | Promise<boolean>,
  ): Condition<MODEL, LAYER> {
    return new Condition<MODEL, LAYER>(async (ctx) => {
      /**
       * Both layers report a failed attempt the same way — a discriminant and
       * the raw error — so the unwrapping reads identically for either.
       */
      const current = (ctx as { current: ModelRetryAttempt<any> }).current;
      if (!isErrorAttempt(current)) return false;
      return predicate(current.error as E, ctx);
    });
  }

  /**
   * Match when the error is an instance of the given error class.
   * Accepts the AI SDK error classes (`APICallError`,
   * `NoObjectGeneratedError`, ...) as well as any `Error` subclass. When
   * the class exposes a static `isInstance` marker check (as the AI SDK
   * classes do), it is preferred over `instanceof`, so matching survives
   * across realms and duplicate package installs.
   *
   * **Important:** returns a `Condition`, not a retryable. Call
   * `.switch()` or `.retry()` to plug it into `retries: [...]`.
   *
   * @example
   * error.isInstance(APICallError).retry({ delay: 1000 })
   * error.isInstance(TypeError).switch({ model: fallback })
   */
  error.isInstance = function isInstance<MODEL extends BOUND = BOUND>(
    cls: ErrorClass,
  ): Condition<MODEL, LAYER> {
    return error<MODEL>((e) =>
      typeof cls.isInstance === 'function'
        ? cls.isInstance(e)
        : e instanceof cls,
    );
  };

  /**
   * Match when the error explicitly carries `isRetryable === flag`.
   *
   * **Important:** returns a `Condition`, not a retryable. Call
   * `.switch()` or `.retry()` to plug it into `retries: [...]`.
   *
   * @example
   * error.isRetryable(true).retry({ delay: 1000 })
   * error.isRetryable(false).switch({ model: fallback })
   */
  error.isRetryable = function isRetryable<MODEL extends BOUND = BOUND>(
    flag = true,
  ): Condition<MODEL, LAYER> {
    return error<MODEL>(
      (e) => APICallError.isInstance(e) && e.isRetryable === flag,
    );
  };

  /**
   * Match by HTTP status code. Numbers match exactly; regular expressions
   * match against the stringified code, useful for range checks.
   *
   * **Important:** returns a `Condition`, not a retryable. Call
   * `.switch()` or `.retry()` to plug it into `retries: [...]`.
   *
   * @example
   * error.statusCode(429, 503).retry({ delay: 1000 })
   * error.statusCode(/^5\d\d$/).switch({ model: fallback })
   */
  error.statusCode = function statusCode<MODEL extends BOUND = BOUND>(
    ...patterns: Array<number | RegExp>
  ): Condition<MODEL, LAYER> {
    return error<MODEL>((e) => {
      if (!APICallError.isInstance(e)) return false;
      const code = e.statusCode;
      if (code === undefined) return false;
      return patterns.some((p) =>
        typeof p === 'number' ? p === code : p.test(String(code)),
      );
    });
  };

  /**
   * Match the error message against substrings or regular expressions.
   * Substring matching is case-insensitive: both the pattern and the
   * message are lowercased before matching. Regular expressions match
   * as written; use the `i` flag for case-insensitive regex matching.
   *
   * **Important:** returns a `Condition`, not a retryable. Call
   * `.switch()` or `.retry()` to plug it into `retries: [...]`.
   *
   * @example
   * error.message('overloaded').switch({ model: fallback })
   * error.message(/rate.?limit/i).retry({ delay: 1000 })
   */
  error.message = function message<MODEL extends BOUND = BOUND>(
    ...patterns: Array<string | RegExp>
  ): Condition<MODEL, LAYER> {
    return error<MODEL>((e) => {
      if (!(e instanceof Error)) return false;
      const lower = e.message.toLowerCase();
      return patterns.some((p) =>
        typeof p === 'string'
          ? lower.includes(p.toLowerCase())
          : p.test(e.message),
      );
    });
  };

  /**
   * Match a timeout error: an `Error` with `name === 'TimeoutError'`,
   * which `AbortSignal.timeout()` produces when the timeout fires.
   * Distinct from `error.isAbort()`, which matches manual aborts.
   *
   * **Important:** returns a `Condition`, not a retryable. Call
   * `.switch()` or `.retry()` to plug it into `retries: [...]`.
   *
   * @example
   * error.isTimeout().switch({ model: fallback, timeout: 60_000 })
   */
  error.isTimeout = function isTimeout<
    MODEL extends BOUND = BOUND,
  >(): Condition<MODEL, LAYER> {
    return error<MODEL>((e) => isTimeoutError(e));
  };

  /**
   * Match a manual abort: an `Error` with `name === 'AbortError'`, which
   * `controller.abort()` produces. Distinct from `error.isTimeout()`,
   * which matches `AbortSignal.timeout()` firing.
   *
   * **Important:** returns a `Condition`, not a retryable. Call
   * `.switch()` or `.retry()` to plug it into `retries: [...]`.
   *
   * @example
   * error.isAbort().switch({ model: fallback })
   */
  error.isAbort = function isAbort<MODEL extends BOUND = BOUND>(): Condition<
    MODEL,
    LAYER
  > {
    return error<MODEL>((e) => isAbortError(e));
  };

  /**
   * Match an `APICallError` by status code, message substring, or regular
   * expression. Numbers match the status code; strings match the message;
   * regular expressions match either the stringified status code or the
   * message. Mix any combination in a single call; matches when any
   * pattern matches.
   *
   * **Important:** returns a `Condition`, not a retryable. Call
   * `.switch()` or `.retry()` to plug it into `retries: [...]`.
   *
   * @example
   * httpStatus(529).switch({ model: fallback })
   * httpStatus(529, 'overloaded').retry({ delay: 1000 })
   */
  function httpStatus<MODEL extends BOUND = BOUND>(
    ...patterns: Array<StatusPattern>
  ): Condition<MODEL, LAYER> {
    const numbers = patterns.filter((p): p is number => typeof p === 'number');
    const strings = patterns.filter((p): p is string => typeof p === 'string');
    const regexes = patterns.filter((p): p is RegExp => p instanceof RegExp);

    const conditions: Array<Condition<MODEL, LAYER>> = [];
    if (numbers.length || regexes.length) {
      conditions.push(error.statusCode<MODEL>(...numbers, ...regexes));
    }
    if (strings.length || regexes.length) {
      conditions.push(error.message<MODEL>(...strings, ...regexes));
    }
    return or(...conditions);
  }

  /**
   * Match a timeout error: an `Error` with `name === 'TimeoutError'`,
   * which `AbortSignal.timeout()` produces when the timeout fires.
   * Distinct from `aborted()`, which matches manual aborts. Convenience
   * wrapper around `error.isTimeout()`.
   *
   * **Important:** returns a `Condition`, not a retryable. Call
   * `.switch()` or `.retry()` to plug it into `retries: [...]`.
   *
   * @example
   * timeout().switch({ model: fallback, timeout: 60_000 })
   * timeout().retry({ delay: 1000 })
   */
  function timeout<MODEL extends BOUND = BOUND>(): Condition<MODEL, LAYER> {
    return error.isTimeout<MODEL>();
  }

  /**
   * Match a manual abort: an `Error` with `name === 'AbortError'`, which
   * `controller.abort()` produces. Distinct from `timeout()`, which
   * matches `AbortSignal.timeout()` firing. Convenience wrapper around
   * `error.isAbort()`.
   *
   * **Important:** returns a `Condition`, not a retryable. Call
   * `.switch()` or `.retry()` to plug it into `retries: [...]`.
   *
   * @example
   * aborted().switch({ model: fallback })
   */
  function aborted<MODEL extends BOUND = BOUND>(): Condition<MODEL, LAYER> {
    return error.isAbort<MODEL>();
  }

  return { error, httpStatus, timeout, aborted };
}
