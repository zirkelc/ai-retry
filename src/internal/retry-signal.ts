import { delay, isAbortError as isSdkAbortError } from '@ai-sdk/provider-utils';
import type { RetryTimeout } from '../types.js';
import { isTimeoutError } from './guards.js';
import { totalTimeoutMs } from './retry-timeout.js';

/**
 * How the inbound signal of a call applies to one of its retries.
 *
 * Without a retry deadline the inbound signal applies unchanged. With one, the
 * retry's own deadline replaces an inbound `TimeoutError` (`AbortSignal.timeout`
 * used as a wall-clock budget), so only the other aborts, a user cancel above
 * all, still reach the retry.
 *
 * Every decision about a retry and the inbound signal goes through this file:
 * the signal the retry runs with, the backoff wait before it, whether it would
 * die at once, and whether a failure is the call being aborted. Deciding it in
 * one place is what keeps those from disagreeing.
 */
type RetryDeadline = { timeout?: RetryTimeout } | undefined;

const hasOwnDeadline = (retry: RetryDeadline): boolean =>
  totalTimeoutMs(retry?.timeout) !== undefined;

/**
 * Whether what can still cancel the retry has already been aborted. Reads the
 * signal only, so it adds no listener to it.
 *
 * Checked before firing a retry: one that is already cancelled would die at
 * once with the same abort, so callers surface the original error instead of
 * firing a misleading retry against a dead signal. A retry deadline rescues it
 * only from an inbound `TimeoutError`, and only a total budget counts as one,
 * because only that replaces the dead signal. The SDK's finer windows are
 * enforced within a call that never starts.
 */
export function isRetryCancelled(
  base: AbortSignal | undefined,
  retry: RetryDeadline,
): boolean {
  if (!base?.aborted) return false;
  return !(hasOwnDeadline(retry) && isTimeoutError(base.reason));
}

/**
 * The part of the inbound signal that still applies to a retry: what can
 * cancel it, as opposed to its deadline. An inbound signal already aborted
 * with a replaced `TimeoutError` drops out entirely, one aborted for another
 * reason is returned as it is, and a live one is followed for the aborts that
 * still apply.
 *
 * Following a live signal adds a listener to it that stays until it aborts, so
 * this is for a signal that lives as long as the retry. A check reads
 * {@link isRetryCancelled} instead, and a wait uses {@link waitBeforeRetry}.
 */
export function retryCancelSignal(
  base: AbortSignal | undefined,
  retry: RetryDeadline,
): AbortSignal | undefined {
  if (!hasOwnDeadline(retry) || base === undefined) {
    return base;
  }

  if (base.aborted) {
    return isRetryCancelled(base, retry) ? base : undefined;
  }

  const controller = new AbortController();
  base.addEventListener(
    'abort',
    () => {
      if (isTimeoutError(base.reason)) return;
      controller.abort(base.reason);
    },
    { once: true },
  );
  return controller.signal;
}

/**
 * Wait out the backoff delay before a retry. The wait ends early on what can
 * still cancel the retry, not on the failed attempt's signal: a deadline the
 * retry replaces must not end it before it starts. The listener it adds to the
 * inbound signal is removed once the wait is over.
 */
export async function waitBeforeRetry(
  delayMs: number | undefined,
  base: AbortSignal | undefined,
  retry: RetryDeadline,
): Promise<void> {
  if (delayMs === undefined) return;

  if (!hasOwnDeadline(retry) || base === undefined) {
    await delay(delayMs, { abortSignal: base });
    return;
  }

  const controller = new AbortController();
  const onAbort = () => {
    if (isRetryCancelled(base, retry)) controller.abort(base.reason);
  };
  onAbort();
  base.addEventListener('abort', onAbort);
  try {
    await delay(delayMs, { abortSignal: controller.signal });
  } finally {
    base.removeEventListener('abort', onAbort);
  }
}

/**
 * Whether `error` is the call being aborted: an abort error, by the SDK's own
 * check (which counts a `TimeoutError` too), while what can cancel the attempt
 * that failed is aborted. `retry` is that attempt's retry, if any: after the
 * inbound deadline fired, a retry that times out on its own deadline has
 * failed, the call was not aborted.
 *
 * Such an error is passed through as it is, never wrapped or turned into an
 * error part, so every consumer sees the abort it would see without a retry
 * wrapper.
 */
export function isCallAbort(
  error: unknown,
  base: AbortSignal | undefined,
  retry?: RetryDeadline,
): boolean {
  return isRetryCancelled(base, retry) && isSdkAbortError(error);
}
