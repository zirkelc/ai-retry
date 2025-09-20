import type { LanguageModelV2 } from '@ai-sdk/provider';

/**
 * Generate a unique key for a LanguageModelV2 instance.
 */
export const getModelKey = (model: LanguageModelV2): string => {
  return `${model.provider}/${model.modelId}`;
};
