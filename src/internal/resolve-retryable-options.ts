import type {
  EmbeddingModelV2,
  LanguageModelV2,
  RetryableOptions,
} from '../types.js';
import { isModelV2 } from '../utils.js';

/**
 * Helper to resolve `RetryableOptions` from either a model and/or options object.
 * Used to support function overloads in retryables:
 * - `retryable(model)`
 * - `retryable(model, options)`
 * - `retryable(options)`
 */
export function resolveRetryableOptions<
  MODEL extends LanguageModelV2 | EmbeddingModelV2,
>(
  modelOrOptions: MODEL | RetryableOptions<MODEL>,
  options?: RetryableOptions<MODEL>,
): RetryableOptions<MODEL> & { model?: MODEL } {
  if (isModelV2(modelOrOptions)) {
    return {
      ...options,
      model: modelOrOptions,
    };
  }

  return modelOrOptions;
}
