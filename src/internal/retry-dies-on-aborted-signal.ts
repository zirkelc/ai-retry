/**
 * Whether firing the chosen retry would be pointless: the inbound caller signal
 * is already aborted and the retry supplies no fresh deadline (`timeout`), so
 * the retry would die instantly with the same abort. Callers surface the
 * original error instead of firing a misleading retry against a dead signal.
 */
export function retryDiesOnAbortedSignal(
  inboundSignal: AbortSignal | undefined,
  retryModel: { timeout?: number },
): boolean {
  return Boolean(inboundSignal?.aborted) && retryModel.timeout === undefined;
}
