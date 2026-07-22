import type { CommitGate } from './detect-stream-commit.js';

/**
 * Thrown by a {@link refusalGate} when the buffered stream text matches a known
 * refusal phrase. Carries the matched phrase and the buffered text so a retry
 * condition can fail over — e.g. match `error.message('cannot assist')`, or a
 * custom `error<MODEL, RefusalError>(e => e instanceof RefusalError)`.
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
 * Lowercase, collapse internal whitespace, and trim, so phrase matching is
 * robust to delta boundaries splitting mid-word and to casing/spacing drift
 * between the model output and the configured phrases.
 */
const normalize = (text: string): string =>
  text.toLowerCase().replace(/\s+/g, ' ').trim();

/**
 * Options for {@link refusalGate}.
 */
export type RefusalGateOptions = {
  /**
   * Build the error thrown when a full refusal phrase is matched. Defaults to a
   * {@link RefusalError}. Override to throw an error your existing conditions
   * already match.
   */
  onRefusal?: (match: { phrase: string; bufferedText: string }) => Error;
};

/**
 * Build a {@link CommitGate} that withholds the commit for a text stream until
 * it can tell a genuine answer from a canned refusal, matching against known
 * refusal phrases (e.g. `"I'm sorry, but I cannot assist"`).
 *
 * A natural-language refusal arrives as ordinary `text-delta` parts and finishes
 * with `finishReason: 'stop'` — no error, no `content-filter` finish reason — so
 * the default first-delta commit can never fail over from it. This gate buffers
 * the leading text (nothing has reached the caller yet) and, per delta:
 * - if the buffer *matches* a full refusal phrase (as a prefix of the buffer) →
 *   throw, so the attempt fails over to another model;
 * - if the buffer is still a *prefix* of some phrase → `'wait'` for more text;
 * - once the buffer *diverges* from every phrase → `'commit'`; it is a real
 *   answer that merely shared a leading fragment, and only the withheld leading
 *   deltas were delayed.
 *
 * Buffering is bounded by the longest phrase: divergence commits immediately, so
 * the gate never holds back more than a phrase's worth of text.
 */
export function refusalGate(
  phrases: ReadonlyArray<string>,
  options?: RefusalGateOptions,
): CommitGate {
  /** Pre-normalize and pair with the original for error reporting. */
  const entries = phrases
    .map((phrase) => ({ phrase, normalized: normalize(phrase) }))
    .filter((entry) => entry.normalized.length > 0);

  return (bufferedText) => {
    const text = normalize(bufferedText);

    const matched = entries.find((entry) => text.startsWith(entry.normalized));
    if (matched) {
      throw (
        options?.onRefusal?.({ phrase: matched.phrase, bufferedText }) ??
        new RefusalError(matched.phrase, bufferedText)
      );
    }

    /** Still an inconclusive prefix of a phrase: keep buffering. */
    if (entries.some((entry) => entry.normalized.startsWith(text))) {
      return 'wait';
    }

    /** Diverged from every phrase: a real answer. */
    return 'commit';
  };
}
