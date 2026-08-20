/**
 * Calculates the exponential backoff delay with optional jitter.
 */
export function calculateExponentialBackoff(
  baseDelay: number,
  backoffFactor: number = 1,
  attempts: number,
  maxDelay?: number,
  useJitter: boolean = true,
): number {
  const factor = Math.max(backoffFactor, 1);
  const delay = baseDelay * factor ** attempts;
  const cappedDelay = maxDelay !== undefined ? Math.min(delay, maxDelay) : delay;

  if (!useJitter) {
    return cappedDelay;
  }

  // Implement "Full Jitter" to avoid thundering herd problem
  return Math.random() * cappedDelay;
}
