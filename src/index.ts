import { createRetryableModel } from './internal/create-retryable-model.js';

export * from './internal/get-model-key.js';
export * from './types.js';
export { isErrorAttempt, isResultAttempt } from './internal/guards.js';

/**
 * Call-level retries: the AI SDK entry points, each taking its own arguments
 * plus a `retry` field.
 */
export {
  retryableEmbed,
  type RetryableEmbed,
} from './call/embedding-model/functions/embed.js';
export {
  retryableEmbedMany,
  type RetryableEmbedMany,
} from './call/embedding-model/functions/embed-many.js';
export {
  retryableGenerateImage,
  type RetryableGenerateImage,
} from './call/image-model/functions/generate-image.js';
export {
  retryableGenerateText,
  type RetryableGenerateText,
} from './call/language-model/functions/generate-text.js';
export {
  retryableStreamText,
  type RetryableStreamText,
} from './call/language-model/functions/stream-text.js';
export type {
  EmbedInput,
  EmbedManyInput,
  GenerateImageInput,
  GenerateTextInput,
  StreamTextInput,
} from './call/inputs.js';
export type {
  CallSuccessAttempt,
  CallSuccessContext,
  CallRetryArg,
  CallRetryOptions,
} from './call/retry-arg.js';

/**
 * The call-level retry context. A different type from the model-level
 * `ModelRetryContext` on purpose: the two layers see different results and different
 * call arguments, and keeping them distinct is what stops a condition written
 * for one from silently typechecking against the other.
 */
export type {
  CallArgs,
  CallFailureContext,
  CallRetries,
  CallRetryable,
  CallRetryAttempt,
  CallRetryContext,
  CallRetryErrorAttempt,
  CallRetryResultAttempt,
} from './call/types.js';

/**
 * The results a call-level result condition judges — each family's entry points
 * as a union discriminated by `operation` — and the guards that narrow them.
 */
export {
  isEmbedManyResult,
  isEmbedResult,
  isGenerateImageResult,
  isGenerateTextResult,
  isStreamTextResult,
} from './call/guards.js';
export type {
  CallEmbeddingModelResult,
  CallFinishReason,
  CallImageModelResult,
  CallLanguageModelResult,
  CallLanguageModelUsage,
  CallResult,
  EmbedManyResultInfo,
  EmbedResultInfo,
  GenerateImageResultInfo,
  GenerateTextResultInfo,
  StreamTextResultInfo,
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
