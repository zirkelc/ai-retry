/**
 * Composable conditions for `retryableGenerateImage`.
 *
 *   import { error, noImage, result, ... }
 *     from 'ai-retry/generate-image/conditions';
 *
 * `result` works here, where it does not under `createRetryableModel`: a
 * call-level retry holds the entry point's own result, so there is something to
 * judge — `res.images` reads directly off it.
 */

import { createErrorAPI } from '../../../internal/conditions/error.js';
import { createNoImageAPI } from '../../../internal/conditions/no-image.js';
import type { ResolvableImageModel } from '../../../types.js';
import { createCallResultAPI } from '../../conditions/result.js';
import type { GenerateImageCommitResult } from '../types.js';

export { and } from '../../../internal/conditions/and.js';
export { not } from '../../../internal/conditions/not.js';
export { or } from '../../../internal/conditions/or.js';

export const { error, httpStatus, timeout, aborted } = createErrorAPI<
  ResolvableImageModel,
  'call'
>();
export const { noImage } = createNoImageAPI<ResolvableImageModel, 'call'>();

export const { result } = createCallResultAPI<
  ResolvableImageModel,
  GenerateImageCommitResult
>();

export type { GenerateImageCommitResult } from '../types.js';
