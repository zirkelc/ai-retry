import { APICallError } from 'ai';
import type {
  EmbeddingModel,
  LanguageModel,
  ResolvableLanguageModel,
  Retryable,
  RetryableOptions,
} from '../types.js';
import { isErrorAttempt } from '../utils.js';

/**
 * Fallback to a different model if the error is non-retryable.
 */
export function requestNotRetryable<
  MODEL extends ResolvableLanguageModel | EmbeddingModel,
>(
  model: MODEL,
  options?: RetryableOptions<MODEL>,
): Retryable<MODEL extends string ? LanguageModel : MODEL> {
  return ((context) => {
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
  }) as Retryable<MODEL extends string ? LanguageModel : MODEL>;
}
