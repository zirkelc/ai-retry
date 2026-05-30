import { NoImageGeneratedError } from 'ai';
import type {
  ResolvableImageModel,
  Retryable,
  RetryableOptions,
} from '../types.js';
import { isErrorAttempt } from '../internal/guards.js';

/**
 * Fallback to a different model if image generation fails with NoImageGeneratedError.
 *
 * @deprecated Use the composable condition API from
 * `ai-retry/image-model/retryables`:
 * `noImage().switch({ model: m })`.
 * See the [v1 README](https://github.com/zirkelc/ai-retry/blob/v1/README.md)
 * for the old function-style docs.
 */
export function noImageGenerated<MODEL extends ResolvableImageModel>(
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
