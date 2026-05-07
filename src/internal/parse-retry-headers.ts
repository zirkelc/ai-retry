/**
 * Upper bound for `Retry-After` / `Retry-After-Ms` honored by retry actions.
 * Servers can technically request very long delays; cap to keep retries
 * responsive.
 */
export const MAX_RETRY_AFTER_MS = 60_000;

export function parseRetryHeaders(
  headers: Record<string, string> | undefined,
): number | null {
  if (!headers) return null;

  const retryAfterMs = headers['retry-after-ms'];
  if (retryAfterMs) {
    const delayMs = Number.parseFloat(retryAfterMs);
    if (!Number.isNaN(delayMs) && delayMs >= 0) {
      return delayMs;
    }
  }

  const retryAfter = headers['retry-after'];
  if (retryAfter) {
    const seconds = Number.parseFloat(retryAfter);
    if (!Number.isNaN(seconds)) {
      return seconds * 1000;
    }

    const date = Date.parse(retryAfter);
    if (!Number.isNaN(date)) {
      return Math.max(0, date - Date.now());
    }
  }

  return null;
}
