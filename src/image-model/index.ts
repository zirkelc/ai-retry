/**
 * Entry point bound to `ImageModel`. Exports `createRetryable` typed for
 * image models plus all image-model conditions.
 *
 *   import { createRetryable, error, noImage }
 *     from 'ai-retry/image-model';
 */

export { createRetryable } from './create-retryable.js';

export * from './conditions/index.js';
