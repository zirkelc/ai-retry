import { RetryableImageModel } from '../internal/retryable-image-model.js';
import type {
  ImageModel,
  ResolvableImageModel,
  RetryableModelOptions,
} from '../types.js';
import { isModel } from '../internal/guards.js';
import { resolveImageModel } from '../internal/resolve-model.js';

/**
 * Build a retryable image model. Accepts a model instance or a gateway
 * model-id string, which is resolved to an image model instance.
 */
export function createRetryable(
  options: Omit<RetryableModelOptions<ImageModel>, 'model'> & {
    model: ResolvableImageModel;
  },
): ImageModel {
  const model = isModel(options.model)
    ? options.model
    : resolveImageModel(options.model);

  return new RetryableImageModel({ ...options, model });
}
