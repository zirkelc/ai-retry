import type { RetryCallAttempt } from '../call/create-retryable-call.js';

/**
 * The outcome of classifying a single stream part:
 * - `'content'` — the first real output; the attempt has committed.
 * - `'preamble'` — a leading control part; keep reading.
 * - `{ error }` — a recoverable failure (error/abort) before content; fail over.
 */
export type StreamPartClassification =
  | 'content'
  | 'preamble'
  | { error: unknown };

/**
 * Classifies a stream part to decide whether an attempt has committed (produced
 * content), is still in its preamble, or failed before producing content.
 */
export type ClassifyStreamPart = (
  part: unknown,
  attempt: RetryCallAttempt,
) => StreamPartClassification;

/**
 * Part types the AI SDK's `streamText`/`streamObject` `fullStream` emits that
 * represent actual output — the boundary past which an attempt is committed.
 */
const STREAM_TEXT_CONTENT_TYPES = new Set([
  'text-delta',
  'reasoning-delta',
  'tool-call',
  'tool-input-start',
  'tool-input-delta',
  'tool-result',
  'source',
  'file',
  'raw',
]);

/**
 * Default classifier for the AI SDK `fullStream` protocol. An `error` part
 * fails over with its error; an `abort` part (a `streamText`-level deadline)
 * fails over with the attempt's abort reason — the real `TimeoutError`/
 * `AbortError`, so error-based retryables can match it — rather than the part's
 * serialized `reason` string. Output parts commit; everything else is preamble.
 */
export const classifyStreamTextPart: ClassifyStreamPart = (part, attempt) => {
  const type = (part as { type?: unknown }).type;

  if (type === 'error') {
    return { error: (part as { error?: unknown }).error };
  }
  if (type === 'abort') {
    return {
      error: attempt.abortSignal?.reason ?? new Error('stream aborted'),
    };
  }
  if (typeof type === 'string' && STREAM_TEXT_CONTENT_TYPES.has(type)) {
    return 'content';
  }
  return 'preamble';
};

/**
 * Drive a stream up to the point its outcome is known, without consuming the
 * whole thing. Reads parts until the first content part (resolves), a
 * recoverable failure (throws, so the caller can fail over), or end-of-stream
 * with no content (resolves — an empty completion is a valid commit).
 *
 * The reader is cancelled once the outcome is known. The passed stream must be
 * safe to read independently of the caller's own consumption — e.g. a fresh
 * tee, as the AI SDK's `result.fullStream` getter produces on each access — so
 * that reading the leading parts here does not steal them from the consumer.
 */
export async function detectStreamCommit(
  fullStream: ReadableStream<unknown>,
  attempt: RetryCallAttempt,
  classify: ClassifyStreamPart,
): Promise<void> {
  const reader = fullStream.getReader();
  try {
    while (true) {
      const { done, value } = await reader.read();
      if (done) return;

      const classification = classify(value, attempt);
      if (classification === 'content') return;
      if (classification === 'preamble') continue;
      throw classification.error;
    }
  } finally {
    void reader.cancel().catch(() => {});
  }
}
