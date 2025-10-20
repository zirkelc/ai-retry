/**
 * Calculates the exponential backoff delay.
 */
export function calculateExponentialBackoff(
  baseDelay: number,
  backoffFactor: number = 1,
  attempts: number,
): number {
  const factor = Math.max(backoffFactor, 1);
  return baseDelay * factor ** attempts;
}
