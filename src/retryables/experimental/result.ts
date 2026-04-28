import type {
  LanguageModelGenerate,
  ResolvableLanguageModel,
  RetryContext,
} from '../../types.js';
import { isResultAttempt } from '../../utils.js';
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
    res: LanguageModelGenerate,
    ctx: RetryContext<MODEL>,
  ) => boolean | Promise<boolean>,
): Condition<MODEL> {
  return new Condition<MODEL>(async (ctx) => {
    if (!isResultAttempt(ctx.current)) return false;
    return predicate(ctx.current.result, ctx);
  });
}
