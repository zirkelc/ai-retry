import type { LanguageModelV2 } from '@ai-sdk/provider';
import { NoObjectGeneratedError, TypeValidationError } from 'ai';
import type { Retryable, RetryModel } from '../create-retryable-model.js';

export function responseSchemaMismatch(
  input: LanguageModelV2 | RetryModel,
): Retryable {
  return (context) => {
    const { error } = context;
    const model = 'model' in input ? input.model : input;

    if (
      NoObjectGeneratedError.isInstance(error) &&
      error.finishReason === 'stop' &&
      TypeValidationError.isInstance(error.cause)
    ) {
      return { model, maxAttempts: 1 };
    }

    return undefined;
  };
}
