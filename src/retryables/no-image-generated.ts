import { NoImageGeneratedError } from 'ai';
import type { ImageModel, Retryable, RetryableOptions } from '../types.js';
import { isErrorAttempt } from '../utils.js';

/**
 * Fallback to a different model if image generation fails with NoImageGeneratedError.
 */
export function noImageGenerated<MODEL extends ImageModel>(
  model: MODEL,
  options?: RetryableOptions<MODEL>,
): Retryable<MODEL> {
  return (context) => {
    const { current } = context;

    if (isErrorAttempt(current)) {
      const { error } = current;

      if (NoImageGeneratedError.isInstance(error)) {
        return { model, maxAttempts: 1, ...options };
      }
    }

    return undefined;
  };
}
