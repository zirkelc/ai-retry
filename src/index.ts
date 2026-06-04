import { createRetryable as createRetryableAnyModel } from './internal/create-retryable-model.js';

export * from './internal/get-model-key.js';
export * from './types.js';
export { isErrorAttempt, isResultAttempt } from './internal/guards.js';

/**
 * Create a retryable model, auto-detecting the model family (language,
 * embedding, or image) from the base model at runtime.
 *
 * @deprecated Import `createRetryable` from a model-specific entry point
 * instead — it is typed for that family and resolves gateway model-id
 * strings for it:
 *
 * - `ai-retry/language-model`
 * - `ai-retry/embedding-model`
 * - `ai-retry/image-model`
 *
 * The model-specific entry points support gateway strings for every
 * family (base model, fallbacks, and `.switch()` targets); this
 * root export resolves a bare string as a language model only.
 */
export const createRetryable = createRetryableAnyModel;
