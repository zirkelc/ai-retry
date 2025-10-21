import { APICallError } from 'ai';
import { resolveRetryableOptions } from '../internal/resolve-retryable-options.js';
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
 * Retry with the same or a different model if the error is retryable with a delay.
 * Uses the `Retry-After` or `Retry-After-Ms` headers if present.
 * Otherwise uses the specified `delay` with exponential backoff if `backoffFactor` is provided.
 */
export function retryAfterDelay<
  MODEL extends LanguageModelV2 | EmbeddingModelV2,
>(model: MODEL, options?: RetryableOptions<MODEL>): Retryable<MODEL>;
export function retryAfterDelay<
  MODEL extends LanguageModelV2 | EmbeddingModelV2,
>(options: RetryableOptions<MODEL>): Retryable<MODEL>;
export function retryAfterDelay<
  MODEL extends LanguageModelV2 | EmbeddingModelV2,
>(
  modelOrOptions: MODEL | RetryableOptions<MODEL>,
  options?: RetryableOptions<MODEL>,
): Retryable<MODEL> {
  const resolvedOptions = resolveRetryableOptions(modelOrOptions, options);

  return (context) => {
    const { current } = context;

    if (isErrorAttempt(current)) {
      const { error } = current;

      if (APICallError.isInstance(error) && error.isRetryable === true) {
        const model = resolvedOptions.model ?? current.model;

        const headerDelay = parseRetryHeaders(error.responseHeaders);
        if (headerDelay !== null) {
          return {
            model,
            ...resolvedOptions,
            delay: Math.min(headerDelay, MAX_RETRY_AFTER_MS),
          };
        }

        return {
          model,
          ...resolvedOptions,
        };
      }
    }

    return undefined;
  };
}
