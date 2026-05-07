import { type AnyModel, Condition } from './condition.js';

/**
 * Match when any of the given conditions match. Evaluates left to right
 * and stops on the first match.
 *
 * @example
 * or(httpStatus(429), error.message('overloaded'))
 */
export function or<MODEL extends AnyModel>(
  ...conditions: Array<Condition<MODEL>>
): Condition<MODEL> {
  return new Condition<MODEL>(async (ctx) => {
    for (const c of conditions) {
      if (await c.evaluate(ctx)) return true;
    }
    return false;
  });
}
