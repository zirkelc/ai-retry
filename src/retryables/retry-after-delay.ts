import { APICallError } from 'ai';
import { parseRetryHeaders } from '../parse-retry-headers.js';
import type {
  EmbeddingModelV2,
  LanguageModelV2,
  Retryable,
  RetryableOptions,
} from '../types.js';
import { isErrorAttempt } from '../utils.js';

const MAX_RETRY_AFTER_MS = 60_000;

/**
 * Retry the current failed attempt with the same model, if the error is retryable.
 * Uses the `Retry-After` or `Retry-After-Ms` headers if present.
 * Otherwise uses the specified `delay` and `backoffFactor` if provided.
 */
export function retryAfterDelay<
  MODEL extends LanguageModelV2 | EmbeddingModelV2,
>(options: RetryableOptions<MODEL>): Retryable<MODEL> {
  return (context) => {
    const { current } = context;

    if (isErrorAttempt(current)) {
      const { error } = current;

      if (APICallError.isInstance(error) && error.isRetryable === true) {
        const model = current.model;

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
