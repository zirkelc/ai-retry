import { getTotalTimeoutMs, type TimeoutConfiguration, type ToolSet } from 'ai';
import type { RetryTimeout } from '../types.js';

/**
 * Reading and merging the deadline a retry sets for its own attempt.
 *
 * A retry states its deadline in the SDK's own vocabulary, so the two ends of
 * the library need different things from it: an entry point with a `timeout`
 * argument wants the whole configuration merged into what the call already had,
 * while everywhere else there is nothing but an `abortSignal` to enforce it
 * with, and only a total budget can be expressed that way.
 */

/** The object form, with the number shorthand expanded. */
type TimeoutObject = Exclude<TimeoutConfiguration<ToolSet>, number>;

const asObject = (
  timeout: TimeoutConfiguration<ToolSet> | undefined,
): TimeoutObject =>
  timeout === undefined
    ? {}
    : typeof timeout === 'number'
      ? { totalMs: timeout }
      : timeout;

/**
 * Drop keys explicitly set to `undefined`, so a retry that names one window
 * cannot silently erase another that the call had configured.
 */
const defined = (timeout: TimeoutObject): TimeoutObject =>
  Object.fromEntries(
    Object.entries(timeout).filter(([, value]) => value !== undefined),
  ) as TimeoutObject;

/**
 * The total budget a deadline amounts to, or `undefined` when it names no
 * total.
 *
 * This is what the deadline collapses to wherever it has to be enforced through
 * `abortSignal`: below a model, and around the entry points that have no
 * `timeout` argument. The finer windows have no meaning there — `chunkMs`
 * describes a gap between stream chunks that only the SDK's own pipeline can
 * measure — so a retry naming only those carries no deadline at all in those
 * places.
 */
export const totalTimeoutMs = (
  timeout: RetryTimeout | undefined,
): number | undefined => getTotalTimeoutMs(timeout);

/**
 * Merge a retry's deadline into the one the call already carried, key by key,
 * so narrowing one window leaves the others standing.
 *
 * A number on either side is shorthand for `totalMs` and expands before
 * merging, which is what lets a retry narrow `totalMs` alone against a call
 * that configured `firstChunkMs` as well. Per-tool budgets merge one level
 * deeper for the same reason.
 */
export function mergeRetryTimeout(
  base: TimeoutConfiguration<ToolSet> | undefined,
  retry: RetryTimeout,
): TimeoutConfiguration<ToolSet> {
  const baseObject = defined(asObject(base));
  const retryObject = defined(asObject(retry));
  const merged: TimeoutObject = { ...baseObject, ...retryObject };

  if (baseObject.tools !== undefined && retryObject.tools !== undefined) {
    merged.tools = { ...baseObject.tools, ...retryObject.tools };
  }

  return merged;
}
