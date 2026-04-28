import { type AnyModel, Condition } from './condition.js';

/**
 * Match only when all of the given conditions match. Evaluates left to
 * right and stops on the first miss.
 *
 * @example
 * and(httpStatus(429), error.message('overloaded'))
 */
export function and<MODEL extends AnyModel>(
  ...conditions: Array<Condition<MODEL>>
): Condition<MODEL> {
  return new Condition<MODEL>(async (ctx) => {
    for (const c of conditions) {
      if (!(await c.evaluate(ctx))) return false;
    }
    return true;
  });
}
