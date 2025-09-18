import type { LanguageModelV2 } from '@ai-sdk/provider';
import { APICallError, NoObjectGeneratedError } from 'ai';
import type { Retryable, RetryModel } from '../create-retryable-model.js';
import { isObject, isString } from '../utils.js';

/**
 *
 */
export function finishReasonContentFilter(
  input: LanguageModelV2 | RetryModel,
): Retryable {
  return (context) => {
    const { error } = context;
    const model = 'model' in input ? input.model : input;

    if (
      APICallError.isInstance(error) &&
      isObject(error.data) &&
      isObject(error.data.error) &&
      isString(error.data.error.code) &&
      error.data.error.code === 'content_filter'
    ) {
      return { model, maxAttempts: 1 };
    }

    if (
      NoObjectGeneratedError.isInstance(error) &&
      error.finishReason === 'content-filter'
    ) {
      return { model, maxAttempts: 1 };
    }

    return undefined;
  };
}
