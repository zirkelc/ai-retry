import type { LanguageModelV2 } from '@ai-sdk/provider';
import { APICallError } from 'ai';
import type { Retryable, RetryModel } from '../create-retryable-model.js';
import { isErrorAttempt } from '../create-retryable-model.js';

/**
 * Fallback to a different model if the provider returns a HTTP 529 error.
 */
export function serviceOverloaded(
  model: LanguageModelV2,
  options?: Omit<RetryModel, 'model'>,
): Retryable {
  return (context) => {
    const { current } = context;

    if (isErrorAttempt(current)) {
      const { error } = current;

      if (APICallError.isInstance(error) && error.statusCode === 529) {
        return { model, maxAttempts: 1, ...options };
      }
    }

    return undefined;
  };
}
