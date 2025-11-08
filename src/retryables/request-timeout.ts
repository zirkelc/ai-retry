import { isAbortError } from '@ai-sdk/provider-utils';
import type {
  EmbeddingModelV2,
  LanguageModelV2,
  Retryable,
  RetryableOptions,
} from '../types.js';
import { isErrorAttempt } from '../utils.js';

/**
 * Fallback to a different model after a timeout/abort error.
 * Use in combination with the `abortSignal` option.
 * If no timeout is specified, a default of 60 seconds is used.
 */
export function requestTimeout<
  MODEL extends LanguageModelV2 | EmbeddingModelV2,
>(model: MODEL, options?: RetryableOptions<MODEL>): Retryable<MODEL> {
  return (context) => {
    const { current } = context;

    if (isErrorAttempt(current)) {
      /**
       * Fallback to the specified model after a timeout/abort error.
       * Provides a fresh timeout signal for the retry attempt.
       */
      if (isAbortError(current.error)) {
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
