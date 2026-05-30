import { RetryableEmbeddingModel } from '../internal/retryable-embedding-model.js';
import type {
  EmbeddingModel,
  ResolvableEmbeddingModel,
  RetryableModelOptions,
} from '../types.js';
import { isModel } from '../internal/guards.js';
import { resolveEmbeddingModel } from '../internal/resolve-model.js';

/**
 * Build a retryable embedding model. Accepts a model instance or a gateway
 * model-id string, which is resolved to an embedding model instance.
 */
export function createRetryable(
  options: Omit<RetryableModelOptions<EmbeddingModel>, 'model'> & {
    model: ResolvableEmbeddingModel;
  },
): EmbeddingModel {
  const model = isModel(options.model)
    ? options.model
    : resolveEmbeddingModel(options.model);

  return new RetryableEmbeddingModel({ ...options, model });
}
