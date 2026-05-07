/**
 * Experimental entry point bound to `EmbeddingModel`. Re-exports
 * `createRetryable` typed for embedding models plus all embedding-model
 * retryables.
 *
 *   import { createRetryable, error, httpStatus }
 *     from 'ai-retry/experimental/embedding-model';
 */

import { createRetryable as createRetryableBase } from '../../create-retryable-model.js';
import type { EmbeddingModel, RetryableModelOptions } from '../../types.js';

export function createRetryable<MODEL extends EmbeddingModel>(
  options: RetryableModelOptions<MODEL>,
): EmbeddingModel {
  return createRetryableBase(options) as EmbeddingModel;
}

export * from './retryables/index.js';
