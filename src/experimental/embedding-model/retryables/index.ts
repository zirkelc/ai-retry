/**
 * Experimental composable conditions bound to `EmbeddingModel`. For use
 * with `embed` and `embedMany`.
 *
 *   import { error, httpStatus, ... }
 *     from 'ai-retry/experimental/embedding-model/retryables';
 */

import type { EmbeddingModel } from '../../../types.js';
import { createErrorAPI } from '../../internal/error.js';

export const { error, httpStatus, timeout, aborted } =
  createErrorAPI<EmbeddingModel>();
