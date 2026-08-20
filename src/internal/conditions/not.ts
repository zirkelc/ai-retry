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
  COMMIT = unknown,
>(condition: Condition<MODEL, LAYER, COMMIT>): Condition<MODEL, LAYER, COMMIT> {
  return new Condition<MODEL, LAYER, COMMIT>(
    async (ctx) => !(await condition.evaluate(ctx)),
  );
}
