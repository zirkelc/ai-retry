/**
 * Entry point bound to `EmbeddingModel`. Exports `createRetryable` typed
 * for embedding models plus all embedding-model conditions.
 *
 *   import { createRetryable, error, httpStatus }
 *     from 'ai-retry/embedding-model';
 */

export { createRetryable } from './create-retryable.js';

export * from './conditions/index.js';
