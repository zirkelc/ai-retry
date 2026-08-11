import type { RetryTimeout } from '../types.js';
import { totalTimeoutMs } from './retry-timeout.js';

/**
 * Whether firing the chosen retry would be pointless: the inbound caller signal
 * is already aborted and the retry supplies no fresh deadline (`timeout`), so
 * the retry would die instantly with the same abort. Callers surface the
 * original error instead of firing a misleading retry against a dead signal.
 *
 * Only a total budget counts as a fresh deadline here, because only that
 * replaces the dead signal. The SDK's finer windows are enforced within a call
 * that never starts.
 */
export function retryDiesOnAbortedSignal(
  inboundSignal: AbortSignal | undefined,
  retryModel: { timeout?: RetryTimeout },
): boolean {
  return (
    Boolean(inboundSignal?.aborted) &&
    totalTimeoutMs(retryModel.timeout) === undefined
  );
}
