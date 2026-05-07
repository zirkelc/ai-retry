import type {
  LanguageModelResult,
  ResolvableLanguageModel,
  RetryContext,
} from '../../types.js';
import { isResultAttempt } from '../../internal/guards.js';
import { Condition } from './condition.js';

/**
 * Build a condition from a predicate over the current generate result.
 * Available for language models only. The predicate runs only when the
 * current attempt succeeded; error attempts return false.
 *
 * @example
 * result<MODEL>((res) => res.finishReason.unified === 'length')
 */
export function result<
  MODEL extends ResolvableLanguageModel = ResolvableLanguageModel,
>(
  predicate: (
    res: LanguageModelResult,
    ctx: RetryContext<MODEL>,
  ) => boolean | Promise<boolean>,
): Condition<MODEL> {
  return new Condition<MODEL>(async (ctx) => {
    if (!isResultAttempt(ctx.current)) return false;
    return predicate(ctx.current.result, ctx);
  });
}

/**
 * The unified finish reason produced by the AI SDK.
 */
export type FinishReason = LanguageModelResult['finishReason']['unified'];

/**
 * Match the result's finish reason against one of the given values.
 *
 * @example
 * result.finishReason('content-filter')
 * result.finishReason('content-filter', 'length')
 */
result.finishReason = function finishReason<
  MODEL extends ResolvableLanguageModel = ResolvableLanguageModel,
>(...reasons: Array<FinishReason>): Condition<MODEL> {
  return result<MODEL>((res) => reasons.includes(res.finishReason.unified));
};
