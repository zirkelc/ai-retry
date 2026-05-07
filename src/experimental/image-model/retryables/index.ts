/**
 * Experimental composable conditions bound to `ImageModel`. For use with
 * `generateImage`.
 *
 *   import { error, noImage, ... }
 *     from 'ai-retry/experimental/image-model/retryables';
 */

import type { ImageModel } from '../../../types.js';
import { createErrorAPI } from '../../internal/error.js';

export const { error, httpStatus, timeout, aborted } =
  createErrorAPI<ImageModel>();
export { noImage } from '../../internal/no-image.js';
