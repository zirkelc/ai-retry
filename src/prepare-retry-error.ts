import { getErrorMessage } from '@ai-sdk/provider';
import { RetryError } from 'ai';
import type {
  EmbeddingModelV2,
  LanguageModelV2,
  RetryAttempt,
} from './types.js';
import { isErrorAttempt } from './utils.js';

export function prepareRetryError<
  MODEL extends LanguageModelV2 | EmbeddingModelV2,
>(error: unknown, attempts: Array<RetryAttempt<MODEL>>) {
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
