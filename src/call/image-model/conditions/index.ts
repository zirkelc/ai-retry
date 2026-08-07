/**
 * Composable conditions bound to `ImageModel`, for the call-level retry
 * function `retryableGenerateImage`.
 *
 *   import { error, noImage, result, ... }
 *     from 'ai-retry/call/image-model/conditions';
 *
 * `result` works here, where it does not under `createRetryableModel`: a
 * call-level retry holds the entry point's own result, so there is something to
 * judge. The image family has a single entry point, so nothing needs narrowing —
 * `res.images` reads directly.
 */

import { createErrorAPI } from '../../../internal/conditions/error.js';
import { createNoImageAPI } from '../../../internal/conditions/no-image.js';
import type { ResolvableImageModel } from '../../../types.js';
import { createCallResultAPI } from '../../conditions/result.js';

export { and } from '../../../internal/conditions/and.js';
export { not } from '../../../internal/conditions/not.js';
export { or } from '../../../internal/conditions/or.js';

export const { error, httpStatus, timeout, aborted } = createErrorAPI<
  ResolvableImageModel,
  'call'
>();
export const { noImage } = createNoImageAPI<ResolvableImageModel, 'call'>();
export const { result } = createCallResultAPI<ResolvableImageModel>();

export { isGenerateImageResult } from '../../guards.js';
export type {
  CallImageModelResult,
  GenerateImageResultInfo,
} from '../../types.js';
