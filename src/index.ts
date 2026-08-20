import { createRetryableModel } from './internal/create-retryable-model.js';

export * from './internal/get-model-key.js';
export * from './types.js';
export { isErrorAttempt, isResultAttempt } from './internal/guards.js';

/**
 * Call-level retries: the AI SDK entry points, each taking its own arguments
 * plus a `retry` field.
 *
 * Each is also published on its own, together with the conditions written
 * against its result — `ai-retry/generate-text` and
 * `ai-retry/generate-text/conditions`, and so on. Prefer those: the conditions
 * only exist there, so importing the function from the same place keeps a call
 * and the retries it is configured with in one import.
 *
 * @deprecated Import from `ai-retry/generate-text`.
 */
export {
  retryableGenerateText as experimental_retryableGenerateText,
  type RetryableGenerateText,
} from './call/generate-text/generate-text.js';
/** @deprecated Import from `ai-retry/stream-text`. */
export {
  retryableStreamText as experimental_retryableStreamText,
  type RetryableStreamText,
} from './call/stream-text/stream-text.js';
/** @deprecated Import from `ai-retry/embed`. */
export {
  retryableEmbed as experimental_retryableEmbed,
  type RetryableEmbed,
} from './call/embed/embed.js';
/** @deprecated Import from `ai-retry/embed-many`. */
export {
  retryableEmbedMany as experimental_retryableEmbedMany,
  type RetryableEmbedMany,
} from './call/embed-many/embed-many.js';
/** @deprecated Import from `ai-retry/generate-image`. */
export {
  retryableGenerateImage as experimental_retryableGenerateImage,
  type RetryableGenerateImage,
} from './call/generate-image/generate-image.js';

/**
 * The arguments a retry may override, per entry point.
 *
 * @deprecated Import each from its own entry point — `GenerateTextInput` from
 * `ai-retry/generate-text`, and so on.
 */
export type {
  EmbedInput,
  EmbedManyInput,
  GenerateImageInput,
  GenerateTextInput,
  StreamTextInput,
} from './call/inputs.js';

/**
 * What each entry point's result conditions judge: the outcome as it stands at
 * the moment the attempt would commit.
 *
 * @deprecated Import each from its own entry point — `GenerateTextCommitResult`
 * from `ai-retry/generate-text`, and so on.
 */
export type { GenerateTextCommitResult } from './call/generate-text/types.js';
/** @deprecated Import from `ai-retry/stream-text`. */
export type { StreamTextCommitResult } from './call/stream-text/types.js';
/** @deprecated Import from `ai-retry/embed`. */
export type { EmbedCommitResult } from './call/embed/types.js';
/** @deprecated Import from `ai-retry/embed-many`. */
export type { EmbedManyCommitResult } from './call/embed-many/types.js';
/** @deprecated Import from `ai-retry/generate-image`. */
export type { GenerateImageCommitResult } from './call/generate-image/types.js';

/**
 * The call-level retry context and the shape of the `retry` argument. A
 * different type from the model-level `ModelRetryContext` on purpose: the two
 * layers see different results and different call arguments, and keeping them
 * distinct is what stops a condition written for one from silently typechecking
 * against the other.
 */
export type {
  CallSettledAttempt,
  CallSettledEvent,
  CallSuccessfulAttempt,
  CallRetryArg,
  CallRetryOptions,
} from './call/retry-arg.js';
export type {
  CallArgs,
  CallFinishReason,
  CallLanguageModelUsage,
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
