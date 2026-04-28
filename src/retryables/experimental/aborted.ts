import type { AnyModel, Condition } from './condition.js';
import { error } from './error.js';

/**
 * Match a manual abort: an `Error` with `name === 'AbortError'`, which
 * `controller.abort()` produces. Distinct from `timeout()`, which
 * matches `AbortSignal.timeout()` firing.
 *
 * @example
 * aborted().switch({ model: fallback })
 */
export function aborted<MODEL extends AnyModel = AnyModel>(): Condition<MODEL> {
  return error<MODEL>(
    (err) => err instanceof Error && err.name === 'AbortError',
  );
}
