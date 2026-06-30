import { gateway } from 'ai';
import type {
  AnyResolvableModel,
  EmbeddingModel,
  ImageModel,
  LanguageModel,
  ResolvedModel,
} from '../types.js';
import { isModel } from './guards.js';

/**
 * Resolves a gateway model-id string to a model instance for a specific
 * model family. A bare string is ambiguous across families, so the
 * caller (which knows its family) supplies the matching resolver.
 */
export type GatewayResolver = (
  id: string,
) => LanguageModel | EmbeddingModel | ImageModel;

export const resolveLanguageModel = (id: string): LanguageModel =>
  gateway.languageModel(id);
export const resolveEmbeddingModel = (id: string): EmbeddingModel =>
  gateway.embeddingModel(id);
export const resolveImageModel = (id: string): ImageModel =>
  gateway.imageModel(id);

/**
 * Resolve a model string via the AI SDK Gateway to a model instance.
 * Instances pass through unchanged; strings are resolved with the given
 * family resolver (defaults to language for backward compatibility).
 */
export function resolveModel<MODEL extends AnyResolvableModel>(
  model: MODEL,
  resolve: GatewayResolver = resolveLanguageModel,
): ResolvedModel<MODEL> {
  const resolvedModel = isModel(model) ? model : resolve(model);

  return resolvedModel as ResolvedModel<MODEL>;
}
