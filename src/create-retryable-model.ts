import { RetryableEmbeddingModel } from './retryable-embedding-model.js';
import { RetryableLanguageModel } from './retryable-language-model.js';
import type {
  EmbeddingModel,
  LanguageModel,
  RetryableModelOptions,
} from './types.js';

export function createRetryable<MODEL extends LanguageModel>(
  options: RetryableModelOptions<MODEL>,
): LanguageModel;
export function createRetryable<MODEL extends EmbeddingModel>(
  options: RetryableModelOptions<MODEL>,
): EmbeddingModel;
export function createRetryable(
  options:
    | RetryableModelOptions<LanguageModel>
    | RetryableModelOptions<EmbeddingModel>,
): LanguageModel | EmbeddingModel {
  if ('doEmbed' in options.model) {
    return new RetryableEmbeddingModel(
      options as RetryableModelOptions<EmbeddingModel>,
    );
  }

  return new RetryableLanguageModel(
    options as RetryableModelOptions<LanguageModel>,
  );
}
