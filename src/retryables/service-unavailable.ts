import { APICallError } from 'ai';
import type {
  AnyResolvableModel,
  Retryable,
  RetryableOptions,
} from '../types.js';
import { isErrorAttempt } from '../internal/guards.js';

/**
 * Fallback to a different model if the provider returns a service unavailable error.
 * This retryable handles HTTP status code 503 (Service Unavailable).
 *
 * @deprecated Use the composable condition API from
 * `ai-retry/<family>-model/conditions`:
 * `httpStatus(503).switch({ model: m })`.
 * See the [v1 README](https://github.com/zirkelc/ai-retry/blob/v1/README.md)
 * for the old function-style docs.
 */
export function serviceUnavailable<MODEL extends AnyResolvableModel>(
  model: MODEL,
  options?: RetryableOptions<MODEL>,
): Retryable<MODEL> {
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
