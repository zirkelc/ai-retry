import { gateway } from 'ai';
import type {
  EmbeddingModel,
  ImageModel,
  ResolvableLanguageModel,
  ResolvedModel,
} from './types.js';
import { isModel } from './utils.js';

/**
 * Resolve a model string via the AI SDK Gateway to a model instance
 */
export function resolveModel<
  MODEL extends ResolvableLanguageModel | EmbeddingModel | ImageModel,
>(model: MODEL): ResolvedModel<MODEL> {
  const resolvedModel = isModel(model) ? model : gateway(model);

  return resolvedModel as ResolvedModel<MODEL>;
}
