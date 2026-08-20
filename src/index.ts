import { createRetryableModel } from './internal/create-retryable-model.js';

export * from './internal/get-model-key.js';
export * from './types.js';
export { isErrorAttempt, isResultAttempt } from './internal/guards.js';

/**
 * The call-level retry context and the shape of the `retry` argument. A
 * different type from the model-level `ModelRetryContext` on purpose: the two
 * layers see different results and different call arguments, and keeping them
 * distinct is what stops a condition written for one from silently typechecking
 * against the other.
 *
 * These are the layer's shared types, so they live here; the functions
 * themselves and their per-entry-point types are published only at
 * `ai-retry/<function>`.
 */
export type {
  CallSettledAttempt,
  CallSettledEvent,
  CallSuccessfulAttempt,
  CallRetryArg,
  CallRetryOptions,
} from './call/retry-arg.js';
export type {
  CallRetries,
  CallRetryable,
  CallRetryAttempt,
  CallRetryContext,
  CallRetryErrorAttempt,
  CallRetryResultAttempt,
} from './call/types.js';

/**
 * Create a retryable model, auto-detecting the model family (language,
 * embedding, or image) from the base model at runtime.
 *
 * @deprecated Import `createRetryableModel` from a model-specific entry
 * point instead — it is typed for that family and resolves gateway
 * model-id strings for it:
 *
 * - `ai-retry/language-model`
 * - `ai-retry/embedding-model`
 * - `ai-retry/image-model`
 *
 * The model-specific entry points support gateway strings for every
 * family (base model, fallbacks, and `.switch()` targets); this
 * root export resolves a bare string as a language model only.
 */
export const createRetryable = createRetryableModel;
