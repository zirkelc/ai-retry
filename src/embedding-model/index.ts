/**
 * Entry point bound to `EmbeddingModel`. Exports `createRetryableModel` typed
 * for embedding models plus all embedding-model conditions.
 *
 *   import { createRetryableModel, error, httpStatus }
 *     from 'ai-retry/embedding-model';
 */

export { createRetryableModel } from './create-retryable-model.js';

export * from './conditions/index.js';
