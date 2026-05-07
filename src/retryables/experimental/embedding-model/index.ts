/**
 * Experimental composable conditions bound to `EmbeddingModel`. For use
 * with `embed` and `embedMany`.
 *
 *   import { error, httpStatus, ... }
 *     from 'ai-retry/retryables/experimental/embedding-model';
 */

import type { EmbeddingModel } from '../../../types.js';
import { createErrorAPI } from '../internal/error.js';

export const { error, httpStatus, timeout, aborted } =
  createErrorAPI<EmbeddingModel>();
