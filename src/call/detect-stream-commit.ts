import type { ProviderMetadata } from 'ai';
import { isStreamContentPart } from '../internal/guards.js';
import type { LanguageModelStreamPart } from '../types.js';
import type {
  CallFinishReason,
  CallLanguageModelUsage,
  StreamTextResultInfo,
} from './types.js';
import type { Settled } from './run-retry-loop.js';

/**
 * An `abort` part as emitted by `streamText` when a deadline fires: the reason
 * is serialized to a string (`getErrorMessage`, i.e. `error.toString()`, so
 * `"<name>: <message>"` for an `Error`/`DOMException`).
 */
type AbortPart = { type: 'abort'; reason?: string };

/**
 * The terminal `finish` part, carrying the generation's outcome. Read directly
 * off the stream rather than awaited from the result object, so a contentless
 * stream can be judged without consuming anything the caller still needs.
 */
type FinishPart = {
  type: 'finish';
  finishReason: CallFinishReason;
  totalUsage: CallLanguageModelUsage;
};

/** A per-step `finish-step` part; the only carrier of provider metadata here. */
type FinishStepPart = {
  type: 'finish-step';
  providerMetadata?: ProviderMetadata;
};

/** `"<name>: <message>"` where the name looks like an error class. */
const NAMED_ERROR = /^([A-Za-z]+Error): ([\s\S]*)$/;

/**
 * Recover a matchable `Error` for an `abort` part.
 *
 * A call-level deadline (`timeout.chunkMs`/`stepMs`/`totalMs`) aborts an
 * *internal* controller, not the caller's signal, so no structured reason
 * survives and only the part's serialized `reason` string is left. Reconstruct
 * an `Error` from it, restoring `name` when the string carries one (e.g.
 * `TimeoutError`, `AbortError`), so name-based conditions (`timeout()`,
 * `aborted()`) and message-based ones match exactly as they would against a
 * thrown error.
 *
 * When the caller's own signal aborted (a genuine cancel), its structured
 * `reason` is preferred: that is the real `Error` instance, so `instanceof`
 * checks survive too.
 */
function abortErrorFromPart(
  part: AbortPart,
  callerSignal: AbortSignal | undefined,
): unknown {
  if (callerSignal?.reason !== undefined) return callerSignal.reason;

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
 * Report a stream that finished without content.
 *
 * Everything comes from the parts already walked past, so nothing has to be
 * awaited off the result object — which matters, because those promises only
 * settle once the caller has consumed the stream we must not touch. That is
 * also why what a condition sees is not the SDK's own `StreamTextResult`: every
 * field on that is a promise, and awaiting one would consume the stream.
 */
function resultFromFinish(
  finish: FinishPart,
  providerMetadata: ProviderMetadata | undefined,
): StreamTextResultInfo {
  return {
    operation: 'streamText',
    finishReason: finish.finishReason,
    usage: finish.totalUsage,
    providerMetadata,
  };
}

/**
 * Drive a stream up to the point its outcome is known, without consuming the
 * whole thing. Reads the result's part stream until one of:
 *
 * - the first content part — the attempt has committed and cannot fail over;
 * - an `error` part — throws its error, so the caller can fail over;
 * - an `abort` part (a call-level deadline) — throws a matchable error
 *   ({@link abortErrorFromPart}): the caller's structured abort reason when
 *   their own signal fired, otherwise an `Error` reconstructed from the part's
 *   serialized `reason`, preserving `name` so `timeout()`/`aborted()` still
 *   match a `chunkMs`/`stepMs`/`totalMs` deadline;
 * - end-of-stream with no content — reports the generation for result
 *   conditions to judge (a `content-filter` finish with no output being the
 *   motivating case).
 *
 * The commit boundary is {@link isStreamContentPart}, the same content-part set
 * the AI SDK's `onChunk` fires on. Everything before it is preamble.
 *
 * Because a pre-commit stream has emitted no text and no tool calls *by
 * definition* — either would have committed it — result conditions are
 * effectively finish-reason-shaped here. That ceiling is inherent to streaming,
 * not to this implementation, and the reported result says so by declaring no
 * content at all.
 *
 * The reader is cancelled once the outcome is known. The passed stream must be
 * safe to read independently of the caller's own consumption — a fresh tee, as
 * `streamText`'s `stream` getter produces on each access — so reading the
 * leading parts here does not steal them from the consumer.
 */
export async function detectStreamCommit(
  stream: ReadableStream<unknown>,
  callerSignal: AbortSignal | undefined,
): Promise<Settled<StreamTextResultInfo>> {
  const reader = stream.getReader();
  let finish: FinishPart | undefined;
  let providerMetadata: ProviderMetadata | undefined;

  try {
    while (true) {
      const { done, value } = await reader.read();
      if (done) {
        return finish
          ? {
              type: 'result',
              result: resultFromFinish(finish, providerMetadata),
            }
          : { type: 'committed' };
      }

      const type = (value as { type?: unknown }).type;

      if (type === 'error') {
        throw (value as { error?: unknown }).error;
      }
      if (type === 'abort') {
        throw abortErrorFromPart(value as AbortPart, callerSignal);
      }
      if (type === 'finish') {
        finish = value as FinishPart;
        continue;
      }
      if (type === 'finish-step') {
        providerMetadata =
          (value as FinishStepPart).providerMetadata ?? providerMetadata;
        continue;
      }
      if (isStreamContentPart(value as LanguageModelStreamPart)) {
        return { type: 'committed' };
      }
    }
  } finally {
    void reader.cancel().catch(() => {});
  }
}
