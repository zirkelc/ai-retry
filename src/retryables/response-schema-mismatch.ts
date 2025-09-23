import type { LanguageModelV2 } from '@ai-sdk/provider';
import { NoObjectGeneratedError, TypeValidationError } from 'ai';
import {
  isErrorAttempt,
  type Retryable,
  type RetryModel,
} from '../create-retryable-model.js';

export function responseSchemaMismatch(
  model: LanguageModelV2,
  options?: Omit<RetryModel, 'model'>,
): Retryable {
  return (context) => {
    const { current } = context;

    if (isErrorAttempt(current)) {
      if (
        NoObjectGeneratedError.isInstance(current.error) &&
        current.error.finishReason === 'stop' &&
        TypeValidationError.isInstance(current.error.cause)
      ) {
        return { model, maxAttempts: 1, ...options };
      }
    }

    return undefined;
  };
}
