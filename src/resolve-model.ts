import { gateway } from 'ai';
import type {
  EmbeddingModel,
  ResolvableLanguageModel,
  ResolvedModel,
} from './types.js';
import { isModel } from './utils.js';

/**
 * Resolve a model string via the AI SDK Gateway to a modelinstance
 */
export function resolveModel<
  MODEL extends ResolvableLanguageModel | EmbeddingModel,
>(model: MODEL): ResolvedModel<MODEL> {
  const resolvedModel = isModel(model) ? model : gateway(model);

  return resolvedModel as ResolvedModel<MODEL>;
}
