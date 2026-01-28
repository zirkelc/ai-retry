import type { Reset } from './types.js';

export type ParsedReset =
  | { type: 'requests'; count: number }
  | { type: 'seconds'; count: number };

/**
 * Parses a `Reset` string into a structured object.
 *
 * `'after-request'` is treated as `{ type: 'requests', count: 0 }`,
 * meaning the sticky model expires immediately (default behavior).
 *
 * @example
 * parseReset(`after-request`);      // { type: 'requests', count: 0 }
 * parseReset(`after-5-requests`);   // { type: 'requests', count: 5 }
 * parseReset(`after-30-seconds`);   // { type: 'seconds', count: 30 }
 */
export function parseReset(reset: Reset): ParsedReset {
  if (reset === `after-request`) {
    return { type: `requests`, count: 0 };
  }

  const requestsMatch = reset.match(/^after-(\d+)-requests$/);
  if (requestsMatch) {
    return { type: `requests`, count: Number.parseInt(requestsMatch[1]!, 10) };
  }

  const secondsMatch = reset.match(/^after-(\d+)-seconds$/);
  if (secondsMatch) {
    return { type: `seconds`, count: Number.parseInt(secondsMatch[1]!, 10) };
  }

  throw new Error(`Invalid reset option: ${reset}`);
}
