import type { AnyModel, Condition } from './condition.js';

/**
 * Invert a condition.
 *
 * @example
 * not(error.isRetryable(true))
 */
export function not<MODEL extends AnyModel>(
  condition: Condition<MODEL>,
): Condition<MODEL> {
  return condition.not();
}
