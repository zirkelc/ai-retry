import type { AnyModel, Condition } from './condition.js';
import { error } from './error.js';

/**
 * Match a timeout error: an `Error` with `name === 'TimeoutError'`,
 * which `AbortSignal.timeout()` produces when the timeout fires.
 * Distinct from `aborted()`, which matches manual aborts.
 *
 * @example
 * timeout().switch({ model: fallback, timeout: 60_000 })
 */
export function timeout<MODEL extends AnyModel = AnyModel>(): Condition<MODEL> {
  return error<MODEL>(
    (err) => err instanceof Error && err.name === 'TimeoutError',
  );
}
