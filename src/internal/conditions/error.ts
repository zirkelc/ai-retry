import { APICallError } from 'ai';
import type { RetryContext } from '../../types.js';
import { isErrorAttempt } from '../guards.js';
import { type AnyModel, Condition } from './condition.js';
import { or } from './or.js';

/**
 * A pattern accepted by `httpStatus`. Numbers match the response status
 * code; strings match the error message as a substring; regular
 * expressions match against both the stringified status code and the
 * error message.
 */
export type StatusPattern = number | string | RegExp;

/**
 * Build the error-side condition helpers (`error`, `httpStatus`,
 * `timeout`, `aborted`) bound to a specific model family. Consumed by
 * `language-model.ts` and `image-model.ts` so each entry point exposes
 * helpers whose `MODEL` generic is constrained to the right family.
 */
export function createErrorAPI<BOUND extends AnyModel>() {
  /**
   * Build a condition from a predicate over the current error. The
   * predicate runs only when the current attempt failed with an error;
   * result attempts return false.
   *
   * **Important:** returns a `Condition`, not a `Retryable`. Call
   * `.switch()` or `.retry()` to plug it into `retries: [...]`.
   *
   * @example
   * error<MODEL, APICallError>(
   *   (e) => APICallError.isInstance(e) && e.statusCode === 418,
   * ).switch({ model: fallback })
   */
  function error<MODEL extends BOUND = BOUND, E = unknown>(
    predicate: (err: E, ctx: RetryContext<MODEL>) => boolean | Promise<boolean>,
  ): Condition<MODEL> {
    return new Condition<MODEL>(async (ctx) => {
      if (!isErrorAttempt(ctx.current)) return false;
      return predicate(ctx.current.error as E, ctx);
    });
  }

  /**
   * Match when the error explicitly carries `isRetryable === flag`.
   *
   * **Important:** returns a `Condition`, not a `Retryable`. Call
   * `.switch()` or `.retry()` to plug it into `retries: [...]`.
   *
   * @example
   * error.isRetryable(true).retry({ delay: 1000 })
   * error.isRetryable(false).switch({ model: fallback })
   */
  error.isRetryable = function isRetryable<MODEL extends BOUND = BOUND>(
    flag = true,
  ): Condition<MODEL> {
    return error<MODEL>(
      (e) => APICallError.isInstance(e) && e.isRetryable === flag,
    );
  };

  /**
   * Match by HTTP status code. Numbers match exactly; regular expressions
   * match against the stringified code, useful for range checks.
   *
   * **Important:** returns a `Condition`, not a `Retryable`. Call
   * `.switch()` or `.retry()` to plug it into `retries: [...]`.
   *
   * @example
   * error.statusCode(429, 503).retry({ delay: 1000 })
   * error.statusCode(/^5\d\d$/).switch({ model: fallback })
   */
  error.statusCode = function statusCode<MODEL extends BOUND = BOUND>(
    ...patterns: Array<number | RegExp>
  ): Condition<MODEL> {
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
   * **Important:** returns a `Condition`, not a `Retryable`. Call
   * `.switch()` or `.retry()` to plug it into `retries: [...]`.
   *
   * @example
   * error.message('overloaded').switch({ model: fallback })
   * error.message(/rate.?limit/i).retry({ delay: 1000 })
   */
  error.message = function message<MODEL extends BOUND = BOUND>(
    ...patterns: Array<string | RegExp>
  ): Condition<MODEL> {
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
   * Match an `APICallError` by status code, message substring, or regular
   * expression. Numbers match the status code; strings match the message;
   * regular expressions match either the stringified status code or the
   * message. Mix any combination in a single call; matches when any
   * pattern matches.
   *
   * **Important:** returns a `Condition`, not a `Retryable`. Call
   * `.switch()` or `.retry()` to plug it into `retries: [...]`.
   *
   * @example
   * httpStatus(529).switch({ model: fallback })
   * httpStatus(529, 'overloaded').retry({ delay: 1000 })
   */
  function httpStatus<MODEL extends BOUND = BOUND>(
    ...patterns: Array<StatusPattern>
  ): Condition<MODEL> {
    const numbers = patterns.filter((p): p is number => typeof p === 'number');
    const strings = patterns.filter((p): p is string => typeof p === 'string');
    const regexes = patterns.filter((p): p is RegExp => p instanceof RegExp);

    const conditions: Array<Condition<MODEL>> = [];
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
   * Distinct from `aborted()`, which matches manual aborts.
   *
   * **Important:** returns a `Condition`, not a `Retryable`. Call
   * `.switch()` or `.retry()` to plug it into `retries: [...]`.
   *
   * @example
   * timeout().switch({ model: fallback, timeout: 60_000 })
   * timeout().retry({ delay: 1000 })
   */
  function timeout<MODEL extends BOUND = BOUND>(): Condition<MODEL> {
    return error<MODEL>(
      (err) => err instanceof Error && err.name === 'TimeoutError',
    );
  }

  /**
   * Match a manual abort: an `Error` with `name === 'AbortError'`, which
   * `controller.abort()` produces. Distinct from `timeout()`, which
   * matches `AbortSignal.timeout()` firing.
   *
   * **Important:** returns a `Condition`, not a `Retryable`. Call
   * `.switch()` or `.retry()` to plug it into `retries: [...]`.
   *
   * @example
   * aborted().switch({ model: fallback })
   */
  function aborted<MODEL extends BOUND = BOUND>(): Condition<MODEL> {
    return error<MODEL>(
      (err) => err instanceof Error && err.name === 'AbortError',
    );
  }

  return { error, httpStatus, timeout, aborted };
}
