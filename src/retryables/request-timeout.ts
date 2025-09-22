import type { LanguageModelV2 } from '@ai-sdk/provider';
import { isAbortError } from '@ai-sdk/provider-utils';
import type { Retryable, RetryModel } from '../create-retryable-model.js';

/**
 * Fallback to a different model after a timeout/abort error.
 * Use in combination with the `abortSignal` option in `generateText`.
 */
export function requestTimeout(input: LanguageModelV2 | RetryModel): Retryable {
  return (context) => {
    const { error } = context.current;
    const model = 'model' in input ? input.model : input;

    /**
     * Fallback to the specified model after all retries are exhausted.
     */
    if (isAbortError(error)) {
      return { model, maxAttempts: 1 };
    }

    return undefined;
  };
}
