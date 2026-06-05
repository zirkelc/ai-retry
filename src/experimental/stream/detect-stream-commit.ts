import { isStreamContentPart } from '../../internal/guards.js';
import type { LanguageModelStreamPart } from '../../types.js';
import type { RetryCallAttempt } from '../call/create-retryable-call.js';

/**
 * Drive a stream up to the point its outcome is known, without consuming the
 * whole thing. Reads `fullStream` parts (the AI SDK `streamText`/`streamObject`
 * protocol) until one of:
 * - the first content part — resolves with `undefined`; the attempt has
 *   committed and cannot fail over;
 * - an `error` part — throws its error, so the caller can fail over;
 * - an `abort` part (a `streamText`-level deadline) — throws the attempt's abort
 *   reason, the real `TimeoutError`/`AbortError` so error-based retryables can
 *   match it, rather than the part's serialized `reason` string;
 * - a `finish` part before any content — resolves with its finish reason, so the
 *   caller can evaluate a *result-based* retry (e.g. a `content-filter` finish
 *   with no content) against the same retryables the model layer uses;
 * - end-of-stream with no content — resolves with `undefined` (an empty
 *   completion is a valid commit).
 *
 * The commit boundary is {@link isStreamContentPart}, the same content-part set
 * the AI SDK's `onChunk` fires on, so call-level and model-level retries stop
 * failing over at exactly the same point. Everything else is preamble.
 *
 * The reader is cancelled once the outcome is known. The passed stream must be
 * safe to read independently of the caller's own consumption — e.g. a fresh
 * tee, as the AI SDK's `result.fullStream` getter produces on each access — so
 * that reading the leading parts here does not steal them from the consumer.
 */
export async function detectStreamCommit(
  fullStream: ReadableStream<unknown>,
  attempt: RetryCallAttempt,
): Promise<string | undefined> {
  const reader = fullStream.getReader();
  try {
    while (true) {
      const { done, value } = await reader.read();
      if (done) return undefined;

      const type = (value as { type?: unknown }).type;

      if (type === 'error') {
        throw (value as { error?: unknown }).error;
      }
      if (type === 'abort') {
        throw attempt.abortSignal?.reason ?? new Error('stream aborted');
      }
      if (isStreamContentPart(value as LanguageModelStreamPart)) {
        return undefined;
      }
      if (type === 'finish') {
        /**
         * `fullStream` (high-level) carries the finish reason as a string; the
         * low-level `{ unified }` shape is tolerated too for non-`streamText`
         * sources.
         */
        const reason = (value as { finishReason?: unknown }).finishReason;
        return typeof reason === 'string'
          ? reason
          : ((reason as { unified?: string } | undefined)?.unified ??
              'unknown');
      }
    }
  } finally {
    void reader.cancel().catch(() => {});
  }
}
