/**
 * Composable conditions bound to `EmbeddingModel`, for the call-level retry
 * functions `retryableEmbed` and `retryableEmbedMany`.
 *
 *   import { error, httpStatus, result, isEmbedResult, ... }
 *     from 'ai-retry/call/embedding-model/conditions';
 *
 * `result` works here, where it does not under `createRetryableModel`: a
 * call-level retry holds the entry point's own result, so there is something to
 * judge. One value or many is the discriminant — narrow with `isEmbedResult` /
 * `isEmbedManyResult` before reading the embeddings.
 */

import { createErrorAPI } from '../../../internal/conditions/error.js';
import type { ResolvableEmbeddingModel } from '../../../types.js';
import { createCallResultAPI } from '../../conditions/result.js';

export { and } from '../../../internal/conditions/and.js';
export { not } from '../../../internal/conditions/not.js';
export { or } from '../../../internal/conditions/or.js';

export const { error, httpStatus, timeout, aborted } = createErrorAPI<
  ResolvableEmbeddingModel,
  'call'
>();
export const { result } = createCallResultAPI<ResolvableEmbeddingModel>();

export { isEmbedManyResult, isEmbedResult } from '../../guards.js';
export type {
  CallEmbeddingModelResult,
  EmbedManyResultInfo,
  EmbedResultInfo,
} from '../../types.js';
