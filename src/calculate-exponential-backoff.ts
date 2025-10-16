/**
 * Calculates the exponential backoff delay.
 */
export function calculateExponentialBackoff(
  baseDelay: number,
  backoffFactor: number,
  attempts: number,
): number {
  return baseDelay * backoffFactor ** attempts;
}
