import { APICallError } from 'ai';
import type {
  EmbeddingModel,
  LanguageModel,
  Retryable,
  RetryableOptions,
} from '../types.js';
import { isErrorAttempt } from '../utils.js';

/**
 * Fallback to a different model if the provider returns a service unavailable error.
 * This retryable handles HTTP status code 503 (Service Unavailable).
 */
export function serviceUnavailable<
  MODEL extends LanguageModel | EmbeddingModel,
>(model: MODEL, options?: RetryableOptions<MODEL>): Retryable<MODEL> {
  return (context) => {
    const { current } = context;

    if (isErrorAttempt(current)) {
      const { error } = current;

      if (APICallError.isInstance(error) && error.statusCode === 503) {
        return { model, maxAttempts: 1, ...options };
      }
    }

    return undefined;
  };
}
