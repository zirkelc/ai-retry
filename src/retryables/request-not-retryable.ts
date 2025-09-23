import type { LanguageModelV2 } from '@ai-sdk/provider';
import { APICallError } from 'ai';
import {
  isErrorAttempt,
  type Retryable,
  type RetryModel,
} from '../create-retryable-model.js';

/**
 * Fallback to a different model if the error is non-retryable.
 */
export function requestNotRetryable(
  model: LanguageModelV2,
  options?: Omit<RetryModel, 'model'>,
): Retryable {
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
