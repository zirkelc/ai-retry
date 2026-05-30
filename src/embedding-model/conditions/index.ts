/**
 * Composable conditions bound to `EmbeddingModel`. For use with
 * `embed` and `embedMany`.
 *
 *   import { error, httpStatus, ... }
 *     from 'ai-retry/embedding-model/conditions';
 */

import type { ResolvableEmbeddingModel } from '../../types.js';
import { createErrorAPI } from '../../internal/conditions/error.js';

export { and } from '../../internal/conditions/and.js';
export { not } from '../../internal/conditions/not.js';
export { or } from '../../internal/conditions/or.js';

export const { error, httpStatus, timeout, aborted } =
  createErrorAPI<ResolvableEmbeddingModel>();
