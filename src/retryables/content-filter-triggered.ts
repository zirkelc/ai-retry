import { APICallError } from 'ai';
import type {
  ResolvableLanguageModel,
  Retryable,
  RetryableOptions,
} from '../types.js';
import {
  isErrorAttempt,
  isObject,
  isResultAttempt,
  isString,
} from '../internal/guards.js';

/**
 * Fallback to a different model if the content filter was triggered.
 *
 * @deprecated Use the composable condition API from
 * `ai-retry/language-model/retryables`:
 * `error(/* check e.data.error.code === 'content_filter' *\/)
 *   .or(finishReason('content-filter'))
 *   .switch({ model: m })`.
 * See the [v1 README](https://github.com/zirkelc/ai-retry/blob/v1/README.md)
 * for the old function-style docs.
 */
export function contentFilterTriggered<MODEL extends ResolvableLanguageModel>(
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

      if (result.finishReason.unified === 'content-filter') {
        return { model, maxAttempts: 1, ...options };
      }
    }

    return undefined;
  };
}
