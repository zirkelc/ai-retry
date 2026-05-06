/**
 * Experimental composable conditions bound to `ImageModel`. For use with
 * `generateImage`.
 *
 *   import { error, noImage, ... }
 *     from 'ai-retry/retryables/experimental/image-model';
 */

import type { ImageModel } from '../../../types.js';
import { createErrorAPI } from '../internal/create-error-api.js';

export const { error, httpStatus, timeout, aborted } =
  createErrorAPI<ImageModel>();
export { noImage } from '../internal/no-image.js';
