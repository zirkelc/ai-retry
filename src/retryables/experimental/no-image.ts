import { NoImageGeneratedError } from 'ai';
import type { ImageModel } from '../../types.js';
import type { Condition } from './condition.js';
import { error } from './error.js';

/**
 * Match when image generation produced no images
 * (`NoImageGeneratedError`).
 *
 * @example
 * noImage().switch({ model: fallback })
 */
export function noImage<
  MODEL extends ImageModel = ImageModel,
>(): Condition<MODEL> {
  return error<MODEL>((err) => NoImageGeneratedError.isInstance(err));
}
