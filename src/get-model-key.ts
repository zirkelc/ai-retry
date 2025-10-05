import type { EmbeddingModelV2, LanguageModelV2 } from './types.js';

/**
 * Generate a unique key for a LanguageModelV2 instance.
 */
export const getModelKey = (
  model: LanguageModelV2 | EmbeddingModelV2,
): string => {
  return `${model.provider}/${model.modelId}`;
};
