/**
 * Entry point bound to `ImageModel`. Exports `createRetryableModel` typed for
 * image models plus all image-model conditions.
 *
 *   import { createRetryableModel, error, noImage }
 *     from 'ai-retry/image-model';
 */

export { createRetryableModel } from './create-retryable-model.js';

export * from './conditions/index.js';
