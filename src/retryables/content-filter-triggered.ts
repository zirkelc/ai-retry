import type { LanguageModelV2 } from '@ai-sdk/provider';
import { APICallError, NoObjectGeneratedError } from 'ai';
import type { Retryable, RetryModel } from '../create-retryable-model.js';
import { isErrorAttempt, isResultAttempt } from '../create-retryable-model.js';
import { isObject, isString } from '../utils.js';

/**
 * Fallback to a different model if the content filter was triggered.
 */
export function contentFilterTriggered(
  model: LanguageModelV2,
  options?: Omit<RetryModel, 'model'>,
): Retryable {
  return (context) => {
    const { current } = context;

    // Check for content filter in error attempts
    if (isErrorAttempt(current)) {
      const { error } = current;

      if (
        APICallError.isInstance(error) &&
        isObject(error.data) &&
        isObject(error.data.error) &&
        isString(error.data.error.code) &&
        error.data.error.code === 'content_filter'
      ) {
        return { model, maxAttempts: 1, ...options };
      }

      // if (
      //   NoObjectGeneratedError.isInstance(error) &&
      //   error.finishReason === 'content-filter'
      // ) {
      //   return { model, maxAttempts: 1 };
      // }
    }

    // Check for content filter in result attempts
    if (isResultAttempt(current)) {
      const { result } = current;

      if (result.finishReason === 'content-filter') {
        return { model, maxAttempts: 1, ...options };
      }
    }

    return undefined;
  };
}
