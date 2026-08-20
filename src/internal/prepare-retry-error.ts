import { getErrorMessage } from '@ai-sdk/provider';
import { RetryError } from 'ai';

/**
 * The parts of an attempt this reads: whether it failed, and — for one that did
 * not — whatever names its outcome.
 *
 * The finish reason sits on the attempt below a model and on the result around a
 * call, so both places are checked. An operation that has no finish reason at
 * all (an embedding, an image) simply has none to report.
 */
type AttemptLike = {
  type: string;
  error?: unknown;
  result?: unknown;
  finishReason?: unknown;
};

/**
 * Describe a non-error attempt for the `RetryError` it is folded into.
 */
function describeResult(attempt: AttemptLike): string {
  const finishReason =
    attempt.finishReason ??
    (attempt.result as { finishReason?: unknown } | undefined)?.finishReason;

  return finishReason === undefined
    ? 'Result'
    : `Result with finishReason: ${String(finishReason)}`;
}

/**
 * Prepare a RetryError that includes all errors from previous attempts.
 */
export function prepareRetryError(
  error: unknown,
  attempts: ReadonlyArray<AttemptLike>,
) {
  const errorMessage = getErrorMessage(error);
  const errors = attempts.map((a) =>
    a.type === 'error' ? a.error : describeResult(a),
  );

  return new RetryError({
    message: `Failed after ${attempts.length} attempts. Last error: ${errorMessage}`,
    reason: 'maxRetriesExceeded',
    errors,
  });
}
