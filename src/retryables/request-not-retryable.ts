import { APICallError } from 'ai';
import type {
  EmbeddingModelV2,
  LanguageModelV2,
  Retryable,
  RetryModel,
} from '../types.js';
import { isErrorAttempt } from '../utils.js';

/**
 * Fallback to a different model if the error is non-retryable.
 */
export function requestNotRetryable<
  MODEL extends LanguageModelV2 | EmbeddingModelV2,
>(model: MODEL, options?: Omit<RetryModel<MODEL>, 'model'>): Retryable<MODEL> {
  return (context) => {
    const { current } = context;

    if (isErrorAttempt(current)) {
      if (
        APICallError.isInstance(current.error) &&
        current.error.isRetryable === false
      ) {
        return { model, maxAttempts: 1, ...options };
      }
    }

    return undefined;
  };
}
