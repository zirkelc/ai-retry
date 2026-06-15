import { APICallError } from 'ai';
import type {
  AnyResolvableModel,
  Retryable,
  RetryableOptions,
} from '../types.js';
import { isErrorAttempt } from '../internal/guards.js';

/**
 * Fallback to a different model if the error is non-retryable.
 *
 * @deprecated Use the composable condition API from
 * `ai-retry/<family>-model/conditions`:
 * `error.isRetryable(false).switch({ model: m })`.
 * See the [v1 README](https://github.com/zirkelc/ai-retry/blob/v1/README.md)
 * for the old function-style docs.
 */
export function requestNotRetryable<MODEL extends AnyResolvableModel>(
  model: MODEL,
  options?: RetryableOptions<MODEL>,
): Retryable<MODEL> {
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
