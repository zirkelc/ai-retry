import { APICallError, type LanguageModelV2 } from '@ai-sdk/provider';
import type { Retryable, RetryModel } from '../create-retryable-model.js';

/**
 * Fallback to a different model if the error is non-retryable.
 */
export function requestNotRetryable(
  input: LanguageModelV2 | RetryModel,
): Retryable {
  return (context) => {
    const { error } = context.current;
    const model = 'model' in input ? input.model : input;

    if (APICallError.isInstance(error) && error.isRetryable === false) {
      return { model, maxAttempts: 1 };
    }

    return undefined;
  };
}
