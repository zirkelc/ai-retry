import { RetryableLanguageModel } from '../internal/retryable-language-model.js';
import type {
  LanguageModel,
  ResolvableLanguageModel,
  RetryableModelOptions,
} from '../types.js';
import { isModel } from '../internal/guards.js';
import { resolveLanguageModel } from '../internal/resolve-model.js';

/**
 * Build a retryable language model. Accepts a model instance or a gateway
 * model-id string, which is resolved to a language model instance.
 */
export function createRetryable(
  options: Omit<RetryableModelOptions<LanguageModel>, 'model'> & {
    model: ResolvableLanguageModel;
  },
): LanguageModel {
  const model = isModel(options.model)
    ? options.model
    : resolveLanguageModel(options.model);

  return new RetryableLanguageModel({ ...options, model });
}
