import { APICallError } from 'ai';
import type {
  EmbeddingModel,
  ImageModel,
  ResolvableLanguageModel,
  Retryable,
  RetryableOptions,
} from '../types.js';
import { isErrorAttempt, isObject, isString } from '../utils.js';

/**
 * Fallback to a different model if the provider returns an overloaded error.
 * This retryable handles the following cases:
 * - Response with status code 529
 * - Response with `type: "overloaded_error"`
 * - Response with a `message` containing "overloaded"
 */
export function serviceOverloaded<
  MODEL extends ResolvableLanguageModel | EmbeddingModel | ImageModel,
>(model: MODEL, options?: RetryableOptions<MODEL>): Retryable<MODEL> {
  return (context) => {
    const { current } = context;

    if (isErrorAttempt(current)) {
      const { error } = current;

      if (APICallError.isInstance(error) && error.statusCode === 529) {
        return { model, maxAttempts: 1, ...options };
      }

      if (isObject(error)) {
        if (
          (isString(error.type) && error.type === 'overloaded_error') ||
          (isString(error.message) &&
            error.message.toLowerCase().includes('overloaded'))
        ) {
          return { model, maxAttempts: 1, ...options };
        }
      }
    }

    return undefined;
  };
}
