/**
 * Experimental entry point bound to `LanguageModel`. Re-exports
 * `createRetryable` typed for language models plus all language-model
 * retryables.
 *
 *   import { createRetryable, error, httpStatus, finishReason }
 *     from 'ai-retry/experimental/language-model';
 */

import { createRetryable as createRetryableBase } from '../../create-retryable-model.js';
import type {
  GatewayLanguageModelId,
  LanguageModel,
  RetryableModelOptions,
} from '../../types.js';

export function createRetryable(
  options: Omit<RetryableModelOptions<LanguageModel>, 'model'> & {
    model: GatewayLanguageModelId;
  },
): LanguageModel;
export function createRetryable<MODEL extends LanguageModel>(
  options: RetryableModelOptions<MODEL>,
): LanguageModel;
export function createRetryable(
  options:
    | RetryableModelOptions<LanguageModel>
    | (Omit<RetryableModelOptions<LanguageModel>, 'model'> & {
        model: GatewayLanguageModelId;
      }),
): LanguageModel {
  return createRetryableBase(options as RetryableModelOptions<LanguageModel>);
}

export * from './retryables/index.js';
