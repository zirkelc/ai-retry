import { isStreamContentPart } from '../../internal/guards.js';
import type { LanguageModelStreamPart } from '../../types.js';
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
 * Default classifier for the AI SDK `fullStream` protocol. An `error` part
 * fails over with its error; an `abort` part (a `streamText`-level deadline)
 * fails over with the attempt's abort reason — the real `TimeoutError`/
 * `AbortError`, so error-based retryables can match it — rather than the part's
 * serialized `reason` string. Output parts commit; everything else is preamble.
 *
 * The commit boundary reuses {@link isStreamContentPart}, the same content-part
 * set the AI SDK's `onChunk` fires on — so call-level and model-level retries
 * stop failing over at exactly the same point.
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
  if (isStreamContentPart(part as LanguageModelStreamPart)) {
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
