import { getErrorMessage } from '@ai-sdk/provider';
import { RetryError } from 'ai';
import type { EmbeddingModel, LanguageModel, RetryAttempt } from './types.js';
import { isErrorAttempt } from './utils.js';

/**
 * Prepare a RetryError that includes all errors from previous attempts.
 */
export function prepareRetryError<MODEL extends LanguageModel | EmbeddingModel>(
  error: unknown,
  attempts: Array<RetryAttempt<MODEL>>,
) {
  const errorMessage = getErrorMessage(error);
  const errors = attempts.flatMap((a) =>
    isErrorAttempt(a)
      ? a.error
      : `Result with finishReason: ${a.result.finishReason}`,
  );

  return new RetryError({
    message: `Failed after ${attempts.length} attempts. Last error: ${errorMessage}`,
    reason: 'maxRetriesExceeded',
    errors,
  });
}
