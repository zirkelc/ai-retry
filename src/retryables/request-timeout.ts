import type {
  AnyResolvableModel,
  Retryable,
  RetryableOptions,
} from '../types.js';
import { isErrorAttempt, isTimeoutError } from '../internal/guards.js';

/**
 * Fallback to a different model after a timeout/abort error.
 * Use in combination with the `abortSignal` option.
 * If no timeout is specified, a default of 60 seconds is used.
 *
 * @deprecated Use the composable condition API from
 * `ai-retry/<family>-model/retryables`:
 * `timeout().switch({ model: m, timeout: 60_000 })`.
 * See the [v1 README](https://github.com/zirkelc/ai-retry/blob/v1/README.md)
 * for the old function-style docs.
 */
export function requestTimeout<MODEL extends AnyResolvableModel>(
  model: MODEL,
  options?: RetryableOptions<MODEL>,
): Retryable<MODEL> {
  return (context) => {
    const { current } = context;

    if (isErrorAttempt(current)) {
      /**
       * Fallback to the specified model after a timeout/abort error.
       * Provides a fresh timeout signal for the retry attempt.
       */
      if (isTimeoutError(current.error)) {
        return {
          model,
          maxAttempts: 1,
          timeout: options?.timeout ?? 60000, // Default 60 second timeout for retry
          ...options,
        };
      }
    }

    return undefined;
  };
}
