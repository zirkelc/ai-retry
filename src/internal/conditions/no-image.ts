import { NoImageGeneratedError } from 'ai';
import type { ResolvableImageModel } from '../../types.js';
import { isErrorAttempt } from '../guards.js';
import { Condition } from './condition.js';

/**
 * Match when image generation produced no images
 * (`NoImageGeneratedError`).
 *
 * **Important:** returns a `Condition`, not a `Retryable`. Call
 * `.switch()` or `.retry()` to plug it into `retries: [...]`.
 *
 * @example
 * noImage().switch({ model: fallback })
 * noImage().retry({ maxAttempts: 3 })
 */
export function noImage<
  MODEL extends ResolvableImageModel = ResolvableImageModel,
>(): Condition<MODEL> {
  return new Condition<MODEL>(async (ctx) => {
    if (!isErrorAttempt(ctx.current)) return false;
    return NoImageGeneratedError.isInstance(ctx.current.error);
  });
}
