import { APICallError } from 'ai';
import type { LanguageModelV2, Retryable, RetryableOptions } from '../types.js';
import {
  isErrorAttempt,
  isObject,
  isResultAttempt,
  isString,
} from '../utils.js';

/**
 * Fallback to a different model if the content filter was triggered.
 */
export function contentFilterTriggered<MODEL extends LanguageModelV2>(
  model: MODEL,
  options?: RetryableOptions<MODEL>,
): Retryable<MODEL> {
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
