import type { LanguageModelV2 } from '@ai-sdk/provider';
import { isAbortError } from '@ai-sdk/provider-utils';
import {
  isErrorAttempt,
  type Retryable,
  type RetryModel,
} from '../create-retryable-model.js';

/**
 * Fallback to a different model after a timeout/abort error.
 * Use in combination with the `abortSignal` option in `generateText`.
 */
export function requestTimeout(
  model: LanguageModelV2,
  options?: Omit<RetryModel, 'model'>,
): Retryable {
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
