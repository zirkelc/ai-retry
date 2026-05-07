/**
 * Experimental composable conditions bound to `LanguageModel`. For use
 * with `generateText`, `generateObject`, `streamText`, `streamObject`.
 *
 *   import { error, httpStatus, finishReason, ... }
 *     from 'ai-retry/experimental/language-model/retryables';
 */

import type { LanguageModel } from '../../../types.js';
import { createErrorAPI } from '../../internal/error.js';

export const { error, httpStatus, timeout, aborted } =
  createErrorAPI<LanguageModel>();
export { result } from '../../internal/result.js';
export { finishReason } from '../../internal/finish-reason.js';
export { schemaInvalid } from '../../internal/schema-invalid.js';
