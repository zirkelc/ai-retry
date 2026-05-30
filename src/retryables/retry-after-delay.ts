import { APICallError } from 'ai';
import {
  MAX_RETRY_AFTER_MS,
  parseRetryHeaders,
} from '../internal/parse-retry-headers.js';
import type {
  AnyResolvableModel,
  Retryable,
  RetryableOptions,
} from '../types.js';
import { isErrorAttempt } from '../internal/guards.js';

/**
 * Retry the current failed attempt with the same model, if the error is retryable.
 * Uses the `Retry-After` or `Retry-After-Ms` headers if present.
 * Otherwise uses the specified `delay` and `backoffFactor` if provided.
 *
 * @deprecated Use the composable condition API from
 * `ai-retry/<family>-model/retryables`:
 * `error.isRetryable(true).retry({ delay, backoffFactor })`.
 * See the [v1 README](https://github.com/zirkelc/ai-retry/blob/v1/README.md)
 * for the old function-style docs.
 */
export function retryAfterDelay<MODEL extends AnyResolvableModel>(
  options: RetryableOptions<MODEL>,
): Retryable<MODEL> {
  return (context) => {
    const { current } = context;

    if (isErrorAttempt(current)) {
      const { error } = current;

      if (APICallError.isInstance(error) && error.isRetryable === true) {
        const model = current.model as unknown as MODEL;

        const headerDelay = parseRetryHeaders(error.responseHeaders);
        if (headerDelay !== null) {
          return {
            model,
            ...options,
            delay: Math.min(headerDelay, MAX_RETRY_AFTER_MS),
            backoffFactor: 1, // No backoff when using server-specified delay
          };
        }

        return {
          model,
          ...options,
        };
      }
    }

    return undefined;
  };
}
