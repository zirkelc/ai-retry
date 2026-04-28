import { APICallError } from 'ai';
import type { RetryContext } from '../../types.js';
import { isErrorAttempt } from '../../utils.js';
import { type AnyModel, Condition } from './condition.js';

/**
 * Build a condition from a predicate over the current error. The
 * predicate runs only when the current attempt failed with an error;
 * result attempts return false.
 *
 * @example
 * error<MODEL, APICallError>(
 *   (e) => APICallError.isInstance(e) && e.statusCode === 418,
 * )
 */
export function error<MODEL extends AnyModel = AnyModel, E = unknown>(
  predicate: (err: E, ctx: RetryContext<MODEL>) => boolean | Promise<boolean>,
): Condition<MODEL> {
  return new Condition<MODEL>(async (ctx) => {
    if (!isErrorAttempt(ctx.current)) return false;
    return predicate(ctx.current.error as E, ctx);
  });
}

/**
 * Higher-level matchers for common error fields. Drop down to `error(fn)`
 * for anything not covered.
 *
 * `Error.name` is intentionally not exposed here because the property
 * name would clash with `Function.prototype.name`. Use `timeout()` or
 * `aborted()` for the common cases, or `error(fn)` for custom names.
 */
export namespace error {
  /**
   * Match when the error explicitly carries `isRetryable === flag`.
   *
   * @example
   * error.isRetryable(true).retry({ delay: 1000 })
   * error.isRetryable(false).switch({ model: fallback })
   */
  export function isRetryable<MODEL extends AnyModel = AnyModel>(
    flag = true,
  ): Condition<MODEL> {
    return error<MODEL>(
      (e) => APICallError.isInstance(e) && e.isRetryable === flag,
    );
  }

  /**
   * Match by HTTP status code. Numbers match exactly; regular expressions
   * match against the stringified code, useful for range checks.
   *
   * @example
   * error.statusCode(429, 503)
   * error.statusCode(/^5\d\d$/)
   */
  export function statusCode<MODEL extends AnyModel = AnyModel>(
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
  }

  /**
   * Match the error message against substrings or regular expressions.
   * Substring matching is case-insensitive: both the pattern and the
   * message are lowercased before matching. Regular expressions match
   * as written; use the `i` flag for case-insensitive regex matching.
   *
   * @example
   * error.message('overloaded')
   * error.message(/rate.?limit/i)
   * error.message('overloaded', /rate.?limit/i)
   */
  export function message<MODEL extends AnyModel = AnyModel>(
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
  }
}
