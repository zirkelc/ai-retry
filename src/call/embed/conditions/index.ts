/**
 * Composable conditions for `retryableEmbed`.
 *
 *   import { error, httpStatus, result, ... }
 *     from 'ai-retry/embed/conditions';
 *
 * `result` works here, where it does not under `createRetryableModel`: a
 * call-level retry holds the entry point's own result, so there is something to
 * judge. It is typed against a single embedding — `res.embedding`, not
 * `res.embeddings` — which is also what stops a condition written here from
 * compiling against `retryableEmbedMany`.
 */

import { createErrorAPI } from '../../../internal/conditions/error.js';
import type { ResolvableEmbeddingModel } from '../../../types.js';
import { createCallResultAPI } from '../../conditions/result.js';
import type { EmbedCommitResult } from '../types.js';

export { and } from '../../../internal/conditions/and.js';
export { not } from '../../../internal/conditions/not.js';
export { or } from '../../../internal/conditions/or.js';

export const { error, httpStatus, timeout, aborted } = createErrorAPI<
  ResolvableEmbeddingModel,
  'call'
>();

export const { result } = createCallResultAPI<
  ResolvableEmbeddingModel,
  EmbedCommitResult
>();

export type { EmbedCommitResult } from '../types.js';
