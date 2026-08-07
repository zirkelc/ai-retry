import { NoImageGeneratedError } from 'ai';
import type { ResolvableImageModel, ModelRetryAttempt } from '../../types.js';
import { isErrorAttempt } from '../guards.js';
import { Condition, type RetryLayer } from './condition.js';

/**
 * Build the image-only condition helper bound to a specific layer, so each
 * image-model condition entry point exposes a `noImage` whose context is the
 * one its layer actually produces.
 */
export function createNoImageAPI<
  BOUND extends ResolvableImageModel = ResolvableImageModel,
  LAYER extends RetryLayer = 'model',
>() {
  /**
   * Match when image generation produced no images
   * (`NoImageGeneratedError`).
   *
   * **Important:** returns a `Condition`, not a retryable. Call
   * `.switch()` or `.retry()` to plug it into `retries: [...]`.
   *
   * @example
   * noImage().switch({ model: fallback })
   * noImage().retry({ maxAttempts: 3 })
   */
  function noImage<MODEL extends BOUND = BOUND>(): Condition<MODEL, LAYER> {
    return new Condition<MODEL, LAYER>(async (ctx) => {
      const current = (ctx as { current: ModelRetryAttempt<any> }).current;
      if (!isErrorAttempt(current)) return false;
      return NoImageGeneratedError.isInstance(current.error);
    });
  }

  return { noImage };
}
