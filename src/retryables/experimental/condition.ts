import { APICallError } from 'ai';
import {
  MAX_RETRY_AFTER_MS,
  parseRetryHeaders,
} from '../../parse-retry-headers.js';
import type {
  EmbeddingModel,
  ImageModel,
  ResolvableLanguageModel,
  Retry,
  Retryable,
  RetryContext,
} from '../../types.js';
import { isErrorAttempt } from '../../utils.js';

/**
 * Any model the retryable system supports.
 */
export type AnyModel = ResolvableLanguageModel | EmbeddingModel | ImageModel;

/**
 * Predicate over a `RetryContext`. May be sync or async.
 */
export type Predicate<MODEL extends AnyModel> = (
  ctx: RetryContext<MODEL>,
) => boolean | Promise<boolean>;

/**
 * Argument shape for `Condition.switch`. The target `model` is required;
 * all other `Retry` fields are optional.
 */
export type SwitchTarget<MODEL extends AnyModel> = { model: MODEL } & Omit<
  Retry<MODEL>,
  'model'
>;

/**
 * Argument shape for `Condition.retry`. Same as `Retry` without `model`,
 * since retry reuses the current model.
 */
export type RetryOptions<MODEL extends AnyModel> = Omit<Retry<MODEL>, 'model'>;

/**
 * A predicate over a `RetryContext` paired with two terminal actions
 * (`switch`, `retry`) that turn it into a `Retryable<MODEL>`. Compose
 * conditions with `and`, `or`, `not`.
 *
 * @example
 * const cond = httpStatus(429, 503);
 * cond.switch({ model: fallback });
 * cond.retry({ delay: 1000 });
 */
export class Condition<MODEL extends AnyModel> {
  constructor(private readonly predicate: Predicate<MODEL>) {}

  /**
   * Run the predicate against a context and resolve to a boolean.
   */
  async evaluate(ctx: RetryContext<MODEL>): Promise<boolean> {
    return this.predicate(ctx);
  }

  /**
   * Switch to a different model when the condition matches.
   *
   * @example
   * httpStatus(529).switch({ model: fallback })
   */
  switch(target: SwitchTarget<MODEL>): Retryable<MODEL> {
    return async (ctx) => {
      if (!(await this.evaluate(ctx))) return undefined;
      return { maxAttempts: 1, ...target };
    };
  }

  /**
   * Retry the same model when the condition matches. Honors
   * `Retry-After` and `Retry-After-Ms` response headers when present,
   * capped at 60 seconds, overriding any provided `delay`.
   *
   * @example
   * error.isRetryable(true).retry({ delay: 1000, backoffFactor: 2 })
   */
  retry(options?: RetryOptions<MODEL>): Retryable<MODEL> {
    return async (ctx) => {
      if (!(await this.evaluate(ctx))) return undefined;

      const model = ctx.current.model as unknown as MODEL;

      if (isErrorAttempt(ctx.current)) {
        const { error: err } = ctx.current;
        if (APICallError.isInstance(err)) {
          const headerDelay = parseRetryHeaders(err.responseHeaders);
          if (headerDelay !== null) {
            return {
              model,
              ...options,
              delay: Math.min(headerDelay, MAX_RETRY_AFTER_MS),
              backoffFactor: 1,
            };
          }
        }
      }

      return { model, ...options };
    };
  }

  /**
   * Combine with another condition; matches when both match.
   *
   * @example
   * httpStatus(429).and(error.message('overloaded'))
   */
  and(other: Condition<MODEL>): Condition<MODEL> {
    return new Condition<MODEL>(
      async (ctx) => (await this.evaluate(ctx)) && (await other.evaluate(ctx)),
    );
  }

  /**
   * Combine with another condition; matches when either matches.
   *
   * @example
   * httpStatus(429).or(error.message('overloaded'))
   */
  or(other: Condition<MODEL>): Condition<MODEL> {
    return new Condition<MODEL>(
      async (ctx) => (await this.evaluate(ctx)) || (await other.evaluate(ctx)),
    );
  }

  /**
   * Invert the condition.
   *
   * @example
   * error.isRetryable(true).not()
   */
  not(): Condition<MODEL> {
    return new Condition<MODEL>(async (ctx) => !(await this.evaluate(ctx)));
  }
}
