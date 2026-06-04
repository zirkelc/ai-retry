import { createRetryableModel as createRetryableEmbeddingModel } from '../embedding-model/create-retryable.js';
import { createRetryableModel as createRetryableImageModel } from '../image-model/create-retryable.js';
import { createRetryableModel as createRetryableLanguageModel } from '../language-model/create-retryable.js';
import type {
  EmbeddingModel,
  GatewayLanguageModelId,
  ImageModel,
  LanguageModel,
  RetryableModelOptions,
} from '../types.js';
import { isEmbeddingModel, isImageModel, isModel } from './guards.js';
import { resolveLanguageModel } from './resolve-model.js';

/**
 * Auto-detecting factory: resolves the base model, infers its family, and
 * delegates to the matching family builder. A bare string is ambiguous
 * across families, so it is resolved as a language model — the per-family
 * entry points (`ai-retry/<family>-model`) resolve strings for embedding
 * and image models too.
 */
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
  const model = isModel(options.model)
    ? options.model
    : resolveLanguageModel(options.model);

  if (isEmbeddingModel(model)) {
    return createRetryableEmbeddingModel({
      ...options,
      model,
    } as RetryableModelOptions<EmbeddingModel>);
  }

  if (isImageModel(model)) {
    return createRetryableImageModel({
      ...options,
      model,
    } as RetryableModelOptions<ImageModel>);
  }

  return createRetryableLanguageModel({
    ...options,
    model,
  } as RetryableModelOptions<LanguageModel>);
}
