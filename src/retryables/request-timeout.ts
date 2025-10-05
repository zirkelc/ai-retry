import { isAbortError } from '@ai-sdk/provider-utils';
import type {
  EmbeddingModelV2,
  LanguageModelV2,
  Retryable,
  RetryModel,
} from '../types.js';
import { isErrorAttempt } from '../utils.js';

/**
 * Fallback to a different model after a timeout/abort error.
 * Use in combination with the `abortSignal` option.
 * Works with both `LanguageModelV2` and `EmbeddingModelV2`.
 */
export function requestTimeout<
  MODEL extends LanguageModelV2 | EmbeddingModelV2,
>(model: MODEL, options?: Omit<RetryModel<MODEL>, 'model'>): Retryable<MODEL> {
  return (context) => {
    const { current } = context;

    if (isErrorAttempt(current)) {
      /**
       * Fallback to the specified model after all retries are exhausted.
       */
      if (isAbortError(current.error)) {
        return { model, maxAttempts: 1, ...options };
      }
    }

    return undefined;
  };
}
