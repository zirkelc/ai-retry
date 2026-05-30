/**
 * Composable conditions bound to `ImageModel`. For use with
 * `generateImage`.
 *
 *   import { error, noImage, ... }
 *     from 'ai-retry/image-model/conditions';
 */

import type { ResolvableImageModel } from '../../types.js';
import { createErrorAPI } from '../../internal/conditions/error.js';

export { and } from '../../internal/conditions/and.js';
export { not } from '../../internal/conditions/not.js';
export { or } from '../../internal/conditions/or.js';

export const { error, httpStatus, timeout, aborted } =
  createErrorAPI<ResolvableImageModel>();
export { noImage } from '../../internal/conditions/no-image.js';
