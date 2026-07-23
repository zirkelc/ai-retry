import type {
  LanguageModelStreamPart,
  LanguageModelStreamTransform,
} from '../../types.js';

/**
 * Thrown-in / emitted by {@link refusalTransform} when the buffered stream text
 * matches a known refusal phrase. Carries the matched phrase and the buffered
 * text so a retry condition can fail over — e.g. match
 * `error(e => e instanceof RefusalError)`, or `error.message('cannot assist')`.
 *
 * The name is `'RefusalError'` so it does not collide with `TimeoutError` /
 * `AbortError` matching.
 */
export class RefusalError extends Error {
  /** The refusal phrase (as configured) that the buffered text matched. */
  readonly phrase: string;
  /** The text accumulated from `text-delta` parts up to the match. */
  readonly bufferedText: string;

  constructor(phrase: string, bufferedText: string) {
    super(`Stream produced a refusal: ${JSON.stringify(bufferedText)}`);
    this.name = 'RefusalError';
    this.phrase = phrase;
    this.bufferedText = bufferedText;
  }
}

/**
 * Options for {@link refusalTransform}.
 */
export type RefusalOptions = {
  /**
   * Build the error emitted when a refusal phrase is matched. Defaults to a
   * {@link RefusalError}. Override to emit an error your existing conditions
   * already match.
   */
  onRefusal?: (match: { phrase: string; bufferedText: string }) => Error;
};

/**
 * Lowercase, collapse internal whitespace, and trim, so phrase matching is
 * robust to delta boundaries splitting mid-word and to casing/spacing drift
 * between the model output and the configured phrases.
 */
const normalize = (text: string): string =>
  text.toLowerCase().replace(/\s+/g, ' ').trim();

/**
 * Build a {@link LanguageModelStreamTransform} that turns a canned refusal into
 * a recoverable `error` part, for use as `experimental_transform` on a retryable
 * model (`createRetryable`).
 *
 * It runs *inside* `doStream` (below `streamText`), buffering the leading
 * `text-delta` parts and, per delta:
 * - if the buffered text *matches* a refusal phrase → emits a single `error`
 *   part (a {@link RefusalError} by default) and drops the refusal text. Because
 *   no content was forwarded yet, the model layer treats it exactly like a
 *   provider error before content — so an error-based condition
 *   (`error(e => e instanceof RefusalError).switch({ model })`) fails over to
 *   another model, and plain `streamText` needs no call-layer wrapper;
 * - if the buffer is still a *prefix* of some phrase → holds the deltas back;
 * - once the buffer *diverges* → flushes the held deltas and forwards the rest
 *   untouched.
 *
 * Non-text parts pass through; any held text is flushed before a non-text part
 * (or at stream end), so a real answer is never reordered or lost. Buffering is
 * bounded by the longest phrase. A fresh instance is created per attempt.
 */
export function refusalTransform(
  phrases: ReadonlyArray<string>,
  options?: RefusalOptions,
): LanguageModelStreamTransform {
  /** Pre-normalize and pair with the original for error reporting. */
  const entries = phrases
    .map((phrase) => ({ phrase, normalized: normalize(phrase) }))
    .filter((entry) => entry.normalized.length > 0);

  return () => {
    let buffer = '';
    let decided = false;
    let held: Array<LanguageModelStreamPart> = [];

    const flush = (
      controller: TransformStreamDefaultController<LanguageModelStreamPart>,
    ) => {
      for (const part of held) controller.enqueue(part);
      held = [];
    };

    return new TransformStream<
      LanguageModelStreamPart,
      LanguageModelStreamPart
    >({
      transform(part, controller) {
        if (decided) return controller.enqueue(part);

        if (part.type !== 'text-delta') {
          /** A non-text part while buffering: commit the held text, then it. */
          if (held.length > 0) {
            decided = true;
            flush(controller);
          }
          return controller.enqueue(part);
        }

        buffer += part.delta;
        held.push(part);
        const text = normalize(buffer);

        const matched = entries.find((entry) =>
          text.startsWith(entry.normalized),
        );
        if (matched) {
          decided = true;
          held = []; // drop the refusal text; emit only the error
          controller.enqueue({
            type: 'error',
            error:
              options?.onRefusal?.({
                phrase: matched.phrase,
                bufferedText: buffer,
              }) ?? new RefusalError(matched.phrase, buffer),
          });
          return;
        }

        /** Still an inconclusive prefix of a phrase: keep holding. */
        if (entries.some((entry) => entry.normalized.startsWith(text))) return;

        /** Diverged from every phrase: a real answer. */
        decided = true;
        flush(controller);
      },
      flush(controller) {
        flush(controller);
      },
    });
  };
}
