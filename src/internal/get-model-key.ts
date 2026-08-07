import type { AnyModel } from '../types.js';

/**
 * Generate a unique key for a model instance.
 */
export const getModelKey = (model: AnyModel): string => {
  return `${model.provider}/${model.modelId}`;
};
