import type {
  EmbeddingModel,
  LanguageModel,
  RetryableOptions,
} from '../types.js';
import { isModel } from '../utils.js';

/**
 * Helper to resolve `RetryableOptions` from either a model and/or options object.
 * Used to support function overloads in retryables:
 * - `retryable(model)`
 * - `retryable(model, options)`
 * - `retryable(options)`
 */
export function resolveRetryableOptions<
  MODEL extends LanguageModel | EmbeddingModel,
>(
  modelOrOptions: MODEL | RetryableOptions<MODEL>,
  options?: RetryableOptions<MODEL>,
): RetryableOptions<MODEL> & { model?: MODEL } {
  if (isModel(modelOrOptions)) {
    return {
      ...options,
      model: modelOrOptions,
    };
  }

  return modelOrOptions;
}
