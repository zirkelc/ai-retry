import { Condition, type RetryLayer } from './condition.js';
import type { AnyResolvableModel } from '../../types.js';

/**
 * Match when any of the given conditions match. Evaluates left to right
 * and stops on the first match.
 *
 * Follows whichever layer the given conditions belong to, so one combinator
 * serves both. Mixing layers is not caught here — the result simply belongs to
 * neither layer, and is rejected by whichever `retries` list it is written
 * into.
 *
 * @example
 * or(httpStatus(429), error.message('overloaded'))
 */
export function or<
  MODEL extends AnyResolvableModel,
  LAYER extends RetryLayer = 'model',
>(...conditions: Array<Condition<MODEL, LAYER>>): Condition<MODEL, LAYER> {
  return new Condition<MODEL, LAYER>(async (ctx) => {
    for (const c of conditions) {
      if (await c.evaluate(ctx)) return true;
    }
    return false;
  });
}
