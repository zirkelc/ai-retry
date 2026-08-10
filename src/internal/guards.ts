import type { TextStreamPart, ToolSet } from 'ai';
import type {
  AnyModel,
  EmbeddingModel,
  ImageModel,
  LanguageModel,
  LanguageModelResult,
  LanguageModelStream,
  LanguageModelStreamPart,
  ModelRetryAttempt,
  ModelRetryErrorAttempt,
  ModelRetryResultAttempt,
} from '../types.js';

export const isObject = (value: unknown): value is Record<string, unknown> =>
  typeof value === 'object' && value !== null;

export const isString = (value: unknown): value is string =>
  typeof value === 'string';

export const isModel = (model: unknown): model is AnyModel =>
  isLanguageModel(model) || isEmbeddingModel(model) || isImageModel(model);

export const isLanguageModel = (model: unknown): model is LanguageModel =>
  isObject(model) &&
  'provider' in model &&
  'modelId' in model &&
  'specificationVersion' in model &&
  'doGenerate' in model &&
  'doStream' in model &&
  model.specificationVersion === 'v4';

export const isEmbeddingModel = (model: unknown): model is EmbeddingModel =>
  isObject(model) &&
  'provider' in model &&
  'modelId' in model &&
  'specificationVersion' in model &&
  'doEmbed' in model &&
  model.specificationVersion === 'v4';

export const isImageModel = (model: unknown): model is ImageModel =>
  isObject(model) &&
  'provider' in model &&
  'modelId' in model &&
  'specificationVersion' in model &&
  'doGenerate' in model &&
  model.specificationVersion === 'v4' &&
  !('doStream' in model) &&
  !('doEmbed' in model);

export const isStreamResult = (
  result: LanguageModelResult | LanguageModelStream,
): result is LanguageModelStream => 'stream' in result;

export const isGenerateResult = (
  result: LanguageModelResult | LanguageModelStream,
): result is LanguageModelResult => 'content' in result;

/**
 * Type guard to check if a retry attempt is an error attempt
 */
export function isErrorAttempt(
  attempt: ModelRetryAttempt<any>,
): attempt is ModelRetryErrorAttempt<any> {
  return attempt.type === 'error';
}

/**
 * Type guard to check if a retry attempt is a result attempt
 */
export function isResultAttempt(
  attempt: ModelRetryAttempt<any>,
): attempt is ModelRetryResultAttempt {
  return attempt.type === 'result';
}

/**
 * Whether a stream part is generated model output, as opposed to the framing
 * around it.
 *
 * This is the commit boundary: the first part it accepts is the point past
 * which an attempt belongs to the caller and can no longer be failed over.
 *
 * Both stream vocabularies are accepted, because the boundary is drawn in both
 * places: `LanguageModelStreamPart` below a model, where the provider's parts
 * are what a retryable model sees, and `TextStreamPart` around a call, where
 * the entry point's own stream is. The types are related but not the same, and
 * the parts they share do not all agree on field names.
 *
 * It mirrors the AI SDK's own `isOutputChunk`, which is internal and not
 * exported, so the agreement is pinned by test rather than by import — see
 * `guards.test.ts`, which reads the classification back out of the SDK at
 * runtime instead of restating it.
 *
 * Empty deltas do not count, matching the SDK. A zero-length delta is a
 * heartbeat rather than content, and committing on one would give the boundary
 * away before anything had been generated.
 */
export const isStreamContentPart = (
  part: LanguageModelStreamPart | TextStreamPart<ToolSet>,
): boolean => {
  switch (part.type) {
    case 'text-delta':
    case 'reasoning-delta':
      /** The provider spells the payload `delta`, the SDK `text`. */
      return ('text' in part ? part.text : part.delta).length > 0;
    case 'tool-input-delta':
      /** Spelled `delta` in both. */
      return part.delta.length > 0;
    /**
     * `TextStreamPart` only. A provider stream expresses a tool call as
     * `tool-input-delta`s and does not stream files at all, so these are
     * unreachable below a model and decide nothing there.
     */
    case 'tool-call':
    case 'file':
    case 'reasoning-file':
      return true;
    default:
      return false;
  }
};

/**
 * Check if an error is a user-initiated abort error (manual controller.abort()).
 * This is distinct from TimeoutError which is thrown by AbortSignal.timeout().
 */
export const isAbortError = (error: unknown): boolean =>
  error instanceof Error && error.name === 'AbortError';

/**
 * Check if an error is a timeout error from AbortSignal.timeout().
 * This is distinct from AbortError which is thrown by manual controller.abort().
 */
export const isTimeoutError = (error: unknown): boolean =>
  error instanceof Error && error.name === 'TimeoutError';
