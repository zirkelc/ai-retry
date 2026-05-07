/**
 * Experimental entry point bound to `ImageModel`. Re-exports
 * `createRetryable` typed for image models plus all image-model
 * retryables.
 *
 *   import { createRetryable, error, noImage }
 *     from 'ai-retry/experimental/image-model';
 */

import { createRetryable as createRetryableBase } from '../../create-retryable-model.js';
import type { ImageModel, RetryableModelOptions } from '../../types.js';

export function createRetryable<MODEL extends ImageModel>(
  options: RetryableModelOptions<MODEL>,
): ImageModel {
  return createRetryableBase(options) as ImageModel;
}

export * from './retryables/index.js';
