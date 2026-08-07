import { APICallError } from 'ai';
import {
  MAX_RETRY_AFTER_MS,
  parseRetryHeaders,
} from '../parse-retry-headers.js';
import type { CallRetryable, CallRetryContext } from '../../call/types.js';
import type {
  AnyResolvableModel,
  Retry,
  ModelRetryAttempt,
  ModelRetryable,
  ModelRetryContext,
} from '../../types.js';
import { isErrorAttempt } from '../guards.js';

/**
 * Which retry layer a condition belongs to.
 *
 * - `'model'` — inside `doGenerate`/`doStream`, judging the provider's result
 * - `'call'` — around the entry point, judging the result the caller receives
 *
 * Carried as a tag rather than as the context type itself because the context
 * is a function of `MODEL`, which is only fixed at the individual condition, not
 * at the factory that builds it.
 */
export type RetryLayer = 'model' | 'call';

/** The context a condition of the given layer is evaluated against. */
export type LayerContext<
  LAYER extends RetryLayer,
  MODEL extends AnyResolvableModel,
> = LAYER extends 'call' ? CallRetryContext<MODEL> : ModelRetryContext<MODEL>;

/** The retryable a condition of the given layer produces. */
export type LayerRetryable<
  MODEL extends AnyResolvableModel,
  INPUT,
  LAYER extends RetryLayer,
> = LAYER extends 'call'
  ? CallRetryable<MODEL, INPUT>
  : ModelRetryable<MODEL, INPUT>;

/**
 * Predicate over a retry context. May be sync or async.
 */
export type Predicate<
  MODEL extends AnyResolvableModel,
  LAYER extends RetryLayer = 'model',
> = (ctx: LayerContext<LAYER, MODEL>) => boolean | Promise<boolean>;

/**
 * Argument shape for `Condition.switch`. The target `model` is required;
 * all other `Retry` fields are optional.
 */
export type SwitchTarget<MODEL extends AnyResolvableModel, INPUT = never> = {
  model: MODEL;
} & Omit<Retry<MODEL, INPUT>, 'model'>;

/**
 * Argument shape for `Condition.retry`. Same as `Retry` without `model`,
 * since retry reuses the current model.
 */
export type RetryOptions<
  MODEL extends AnyResolvableModel,
  INPUT = never,
> = Omit<Retry<MODEL, INPUT>, 'model'>;

/**
 * A predicate over a retry context paired with two terminal actions
 * (`switch`, `retry`) that turn it into a retryable. Compose conditions with
 * `and`, `or`, `not`.
 *
 * `LAYER` decides which context the predicate sees and which retryable the
 * terminal actions produce. It defaults to the model layer, so `Condition<MODEL>`
 * keeps meaning exactly what it always did. Because the two layers' contexts are
 * unrelated types, a condition built for one is rejected by the other's
 * `retries` list rather than silently accepted.
 *
 * @example
 * const cond = httpStatus(429, 503);
 * cond.switch({ model: fallback });
 * cond.retry({ delay: 1000 });
 */
export class Condition<
  MODEL extends AnyResolvableModel,
  LAYER extends RetryLayer = 'model',
> {
  constructor(private readonly predicate: Predicate<MODEL, LAYER>) {}

  /**
   * Run the predicate against a context and resolve to a boolean.
   */
  async evaluate(ctx: LayerContext<LAYER, MODEL>): Promise<boolean> {
    return this.predicate(ctx);
  }

  /**
   * Switch to a different model when the condition matches.
   *
   * `options` is left unbound: its shape depends on which API the resulting
   * retryable is handed to (provider-level call options for a retryable model,
   * the entry point's own arguments for a call-level function), and that is
   * not known here. Writing no `options` produces a retryable every API
   * accepts; writing some checks them against wherever it ends up, which is
   * why an override that belongs to a different entry point is reported at the
   * `retries` list rather than here.
   *
   * @example
   * httpStatus(529).switch({ model: fallback })
   */
  switch<INPUT = never>(
    target: SwitchTarget<MODEL, INPUT>,
  ): LayerRetryable<MODEL, INPUT, LAYER> {
    const retryable = async (ctx: LayerContext<LAYER, MODEL>) => {
      if (!(await this.evaluate(ctx))) return undefined;
      return { maxAttempts: 1, ...target };
    };

    return retryable as LayerRetryable<MODEL, INPUT, LAYER>;
  }

  /**
   * Retry the same model when the condition matches. Honors
   * `Retry-After` and `Retry-After-Ms` response headers when present,
   * capped at 60 seconds, overriding any provided `delay`.
   *
   * `maxAttempts` defaults to 2 (one original attempt + one retry).
   * Lower values are rejected: `maxAttempts: 1` would count the original
   * failed attempt against the budget and never actually retry. Use
   * `.switch({ model: ... })` if you want a single attempt against a
   * different model.
   *
   * @example
   * error.isRetryable(true).retry({ delay: 1000, backoffFactor: 2 })
   */
  retry<INPUT = never>(
    options?: RetryOptions<MODEL, INPUT>,
  ): LayerRetryable<MODEL, INPUT, LAYER> {
    if (options?.maxAttempts !== undefined && options.maxAttempts < 2) {
      throw new Error(
        `Condition.retry() requires maxAttempts >= 2 (got ${options.maxAttempts}); use .switch() for a single attempt against a different model.`,
      );
    }

    const retryable = async (ctx: LayerContext<LAYER, MODEL>) => {
      if (!(await this.evaluate(ctx))) return undefined;

      /**
       * Both layers carry the same two fields here — the attempt's kind and the
       * model it ran against — so the decision reads identically for either.
       */
      const current = (ctx as { current: ModelRetryAttempt<any> }).current;
      const model = current.model as unknown as MODEL;

      if (isErrorAttempt(current)) {
        const { error: err } = current;
        if (APICallError.isInstance(err)) {
          const headerDelay = parseRetryHeaders(err.responseHeaders);
          if (headerDelay !== null) {
            return {
              maxAttempts: 2,
              ...options,
              delay: Math.min(headerDelay, MAX_RETRY_AFTER_MS),
              backoffFactor: 1,
              model,
            };
          }
        }
      }

      return { maxAttempts: 2, ...options, model };
    };

    return retryable as LayerRetryable<MODEL, INPUT, LAYER>;
  }
}
