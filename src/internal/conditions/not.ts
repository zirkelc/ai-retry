import { Condition, type RetryLayer } from './condition.js';
import type { AnyResolvableModel } from '../../types.js';

/**
 * Invert a condition. Follows whichever layer the given condition belongs to.
 *
 * @example
 * not(error.isRetryable(true))
 */
export function not<
  MODEL extends AnyResolvableModel,
  LAYER extends RetryLayer = 'model',
>(condition: Condition<MODEL, LAYER>): Condition<MODEL, LAYER> {
  return new Condition<MODEL, LAYER>(
    async (ctx) => !(await condition.evaluate(ctx)),
  );
}
