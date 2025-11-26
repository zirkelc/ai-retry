import { gateway } from 'ai';
import { RetryableEmbeddingModel } from './retryable-embedding-model.js';
import { RetryableLanguageModel } from './retryable-language-model.js';
import type {
  EmbeddingModel,
  GatewayLanguageModelId,
  LanguageModel,
  RetryableModelOptions,
} from './types.js';
import { isEmbeddingModel, isModel } from './utils.js';

export function createRetryable(
  options: Omit<RetryableModelOptions<LanguageModel>, 'model'> & {
    model: GatewayLanguageModelId;
  },
): LanguageModel;
// export function createRetryable<MODEL extends EmbeddingModel>(
//   options: Omit<RetryableModelOptions<MODEL>, 'model'> & { model: GatewayEmbeddingModelId },
// ): EmbeddingModel;
export function createRetryable(
  options: RetryableModelOptions<LanguageModel>,
): LanguageModel;
export function createRetryable(
  options: RetryableModelOptions<EmbeddingModel>,
): EmbeddingModel;
export function createRetryable(
  options:
    | RetryableModelOptions<LanguageModel>
    | RetryableModelOptions<EmbeddingModel>
    // | RetryableModelOptions<EmbeddingModel>
    | (Omit<RetryableModelOptions<LanguageModel>, 'model'> & {
        model: GatewayLanguageModelId;
      }),
  // | Omit<RetryableModelOptions<EmbeddingModel>, 'model'> & { model: GatewayEmbeddingModelId }
): LanguageModel | EmbeddingModel {
  const model = isModel(options.model) ? options.model : gateway(options.model);

  if (isEmbeddingModel(model)) {
    return new RetryableEmbeddingModel({
      ...options,
      model,
    } as RetryableModelOptions<EmbeddingModel>);
  }

  return new RetryableLanguageModel({
    ...options,
    model,
  } as RetryableModelOptions<LanguageModel>);
}
