import { gateway } from 'ai';
import { RetryableEmbeddingModel } from './retryable-embedding-model.js';
import { RetryableImageModel } from './retryable-image-model.js';
import { RetryableLanguageModel } from './retryable-language-model.js';
import type {
  EmbeddingModel,
  GatewayLanguageModelId,
  ImageModel,
  LanguageModel,
  RetryableModelOptions,
} from './types.js';
import { isEmbeddingModel, isImageModel, isModel } from './utils.js';

export function createRetryable<MODEL extends LanguageModel>(
  options: Omit<RetryableModelOptions<LanguageModel>, 'model'> & {
    model: GatewayLanguageModelId;
  },
): LanguageModel;
export function createRetryable<MODEL extends LanguageModel>(
  options: RetryableModelOptions<MODEL>,
): LanguageModel;
export function createRetryable<MODEL extends EmbeddingModel>(
  options: RetryableModelOptions<MODEL>,
): EmbeddingModel;
export function createRetryable<MODEL extends ImageModel>(
  options: RetryableModelOptions<MODEL>,
): ImageModel;
export function createRetryable(
  options:
    | RetryableModelOptions<LanguageModel>
    | RetryableModelOptions<EmbeddingModel>
    | RetryableModelOptions<ImageModel>
    | (Omit<RetryableModelOptions<LanguageModel>, 'model'> & {
        model: GatewayLanguageModelId;
      }),
): LanguageModel | EmbeddingModel | ImageModel {
  const model = isModel(options.model) ? options.model : gateway(options.model);

  if (isEmbeddingModel(model)) {
    return new RetryableEmbeddingModel({
      ...options,
      model,
    } as RetryableModelOptions<EmbeddingModel>);
  }

  if (isImageModel(model)) {
    return new RetryableImageModel({
      ...options,
      model,
    } as RetryableModelOptions<ImageModel>);
  }

  return new RetryableLanguageModel({
    ...options,
    model,
  } as RetryableModelOptions<LanguageModel>);
}
