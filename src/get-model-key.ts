import type { EmbeddingModel, ImageModel, LanguageModel } from './types.js';

/**
 * Generate a unique key for a model instance.
 */
export const getModelKey = (
  model: LanguageModel | EmbeddingModel | ImageModel,
): string => {
  return `${model.provider}/${model.modelId}`;
};
