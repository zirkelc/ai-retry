import { isStreamContentPart } from '../../internal/guards.js';
import type { LanguageModelStreamPart } from '../../types.js';
import type { RetryCallAttempt } from '../call/create-retryable-call.js';

/**
 * An `abort` part as emitted by `streamText`/`streamObject` when a deadline
 * fires: the reason is serialized to a string (`getErrorMessage`, i.e.
 * `error.toString()`, so `"<name>: <message>"` for an `Error`/`DOMException`).
 */
type AbortPart = { type: 'abort'; reason?: string };

/** `"<name>: <message>"` where the name looks like an error class. */
const NAMED_ERROR = /^([A-Za-z]+Error): ([\s\S]*)$/;

/**
 * Recover a matchable `Error` for an `abort` part.
 *
 * A `streamText`-level deadline (`timeout.chunkMs`/`stepMs`/`totalMs`) aborts an
 * *internal* controller, not the attempt's `abortSignal`, so
 * `attempt.abortSignal.reason` is absent and only the part's serialized `reason`
 * string survives. Reconstruct an `Error` from it, restoring `name` when the
 * string carries one (e.g. `TimeoutError`, `AbortError`), so name-based
 * conditions (`timeout()`, `aborted()`) and message-based ones (`error.message`)
 * can match — the same way they would against a thrown error.
 *
 * When the attempt's own signal did abort (a genuine caller cancel or an
 * ai-retry per-attempt deadline), its structured `reason` is preferred: it is
 * the real `Error` instance, so `instanceof` checks survive too.
 */
function abortErrorFromPart(
  part: AbortPart,
  attempt: RetryCallAttempt,
): unknown {
  if (attempt.abortSignal?.reason !== undefined) {
    return attempt.abortSignal.reason;
  }

  const reason = part.reason;
  if (typeof reason !== 'string' || reason.length === 0) {
    return new Error('stream aborted');
  }

  const [, name, message] = NAMED_ERROR.exec(reason) ?? [];
  if (name !== undefined && message !== undefined) {
    const error = new Error(message);
    error.name = name;
    return error;
  }
  return new Error(reason);
}

/**
 * A gate that decides, from the text accumulated so far, whether a
 * text-producing attempt has really committed. Consulted on every `text-delta`
 * with the running concatenation of their `.text`:
 * - returns `'commit'` — the text has diverged from anything worth withholding;
 *   the attempt commits and cannot fail over;
 * - returns `'wait'` — the text is still an inconclusive prefix; keep buffering
 *   and re-consult on the next delta (the caller has not started consuming yet,
 *   so nothing has reached the user);
 * - throws — treat the buffered text as a failure and fail over. The thrown
 *   error is what the retry conditions see, so throw one they can match (e.g. a
 *   message carrying the matched refusal phrase).
 *
 * Only `text-delta` parts pass through the gate; any other content part
 * (`tool-call`, `source`, …) commits immediately, since a text refusal cannot
 * arrive as one. See {@link refusalGate} for the built-in phrase matcher.
 */
export type CommitGate = (bufferedText: string) => 'commit' | 'wait';

/**
 * Drive a stream up to the point its outcome is known, without consuming the
 * whole thing. Reads the result's part stream (the AI SDK `streamText`/
 * `streamObject` protocol — `stream`, or `fullStream` for `streamObject`) until
 * one of:
 * - the first content part — resolves; the attempt has committed and cannot
 *   fail over. When a `gate` is supplied, `text-delta` parts are instead
 *   buffered and the gate decides commit/wait/fail-over (see {@link CommitGate});
 * - an `error` part — throws its error, so the caller can fail over;
 * - an `abort` part (a `streamText`-level deadline) — throws a matchable error
 *   ({@link abortErrorFromPart}): the attempt's structured abort reason when its
 *   own signal fired, otherwise an `Error` reconstructed from the part's
 *   serialized `reason` (preserving `name`, so `timeout()`/`aborted()` still
 *   match a `chunkMs`/`stepMs`/`totalMs` deadline);
 * - end-of-stream with no content — resolves (an empty completion, e.g. a
 *   `content-filter` finish with no output, is a valid commit here; result-based
 *   conditions are handled at the model layer, below `streamText`).
 *
 * Without a `gate` the commit boundary is {@link isStreamContentPart}, the same
 * content-part set the AI SDK's `onChunk` fires on, so call-level and
 * model-level retries stop failing over at exactly the same point. Everything
 * else is preamble. A `gate` moves the text-commit boundary later — past the
 * leading deltas it withholds — but only for text; it never changes when
 * non-text content or an error/abort commits.
 *
 * The reader is cancelled once the outcome is known. The passed stream must be
 * safe to read independently of the caller's own consumption — e.g. a fresh
 * tee, as the AI SDK's `result.stream`/`result.fullStream` getter produces on
 * each access — so reading the leading parts here does not steal them from the
 * consumer.
 */
export async function detectStreamCommit(
  stream: ReadableStream<unknown>,
  attempt: RetryCallAttempt,
  gate?: CommitGate,
): Promise<void> {
  const reader = stream.getReader();
  let bufferedText = '';
  try {
    while (true) {
      const { done, value } = await reader.read();
      if (done) return;

      const type = (value as { type?: unknown }).type;

      if (type === 'error') {
        throw (value as { error?: unknown }).error;
      }
      if (type === 'abort') {
        throw abortErrorFromPart(value as AbortPart, attempt);
      }
      if (gate && type === 'text-delta') {
        bufferedText += (value as { text?: string }).text ?? '';
        /** Gate throws to fail over; 'wait' buffers the next delta. */
        if (gate(bufferedText) === 'commit') return;
        continue;
      }
      if (isStreamContentPart(value as LanguageModelStreamPart)) {
        return;
      }
    }
  } finally {
    void reader.cancel().catch(() => {});
  }
}
