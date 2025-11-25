import type { EmbeddingModel, LanguageModel } from './types.js';

/**
 * Generate a unique key for a LanguageModel instance.
 */
export const getModelKey = (model: LanguageModel | EmbeddingModel): string => {
  return `${model.provider}/${model.modelId}`;
};
