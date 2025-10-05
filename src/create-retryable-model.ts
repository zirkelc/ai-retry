import { RetryableEmbeddingModel } from './retryable-embedding-model.js';
import { RetryableLanguageModel } from './retryable-language-model.js';
import type {
  EmbeddingModelV2,
  LanguageModelV2,
  RetryableModelOptions,
} from './types.js';

export function createRetryable<MODEL extends LanguageModelV2>(
  options: RetryableModelOptions<MODEL>,
): LanguageModelV2;
export function createRetryable<MODEL extends EmbeddingModelV2>(
  options: RetryableModelOptions<MODEL>,
): EmbeddingModelV2;
export function createRetryable(
  options:
    | RetryableModelOptions<LanguageModelV2>
    | RetryableModelOptions<EmbeddingModelV2>,
): LanguageModelV2 | EmbeddingModelV2 {
  if ('doEmbed' in options.model) {
    return new RetryableEmbeddingModel(
      options as RetryableModelOptions<EmbeddingModelV2>,
    );
  }

  return new RetryableLanguageModel(
    options as RetryableModelOptions<LanguageModelV2>,
  );
}
