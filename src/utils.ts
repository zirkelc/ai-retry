import type {
  LanguageModelV2,
  LanguageModelV2StreamPart,
} from '@ai-sdk/provider';
import type {
  EmbeddingModelV2,
  LanguageModelV2Generate,
  LanguageModelV2Stream,
  Retry,
  RetryAttempt,
  RetryErrorAttempt,
  RetryResultAttempt,
} from './types.js';

export const isObject = (value: unknown): value is Record<string, unknown> =>
  typeof value === 'object' && value !== null;

export const isString = (value: unknown): value is string =>
  typeof value === 'string';

export const isModelV2 = (
  model: unknown,
): model is LanguageModelV2 | EmbeddingModelV2 =>
  isLanguageModelV2(model) || isEmbeddingModelV2(model);

export const isLanguageModelV2 = (model: unknown): model is LanguageModelV2 =>
  isObject(model) &&
  'provider' in model &&
  'modelId' in model &&
  'specificationVersion' in model &&
  model.specificationVersion === 'v2';

export const isEmbeddingModelV2 = (model: unknown): model is EmbeddingModelV2 =>
  isObject(model) &&
  'provider' in model &&
  'modelId' in model &&
  'specificationVersion' in model &&
  model.specificationVersion === 'v2';

export const isStreamResult = (
  result: LanguageModelV2Generate | LanguageModelV2Stream,
): result is LanguageModelV2Stream => 'stream' in result;

export const isGenerateResult = (
  result: LanguageModelV2Generate | LanguageModelV2Stream,
): result is LanguageModelV2Generate => 'content' in result;

/**
 * Type guard to check if a retry attempt is an error attempt
 */
export function isErrorAttempt(
  attempt: RetryAttempt<any>,
): attempt is RetryErrorAttempt<any> {
  return attempt.type === 'error';
}

/**
 * Type guard to check if a retry attempt is a result attempt
 */
export function isResultAttempt(
  attempt: RetryAttempt<any>,
): attempt is RetryResultAttempt {
  return attempt.type === 'result';
}

/**
 * Check if a stream part is a content part (e.g., text delta, reasoning delta, source, tool call, tool result).
 * These types are also emitted by `onChunk` callbacks.
 * @see https://github.com/vercel/ai/blob/1fe4bd4144bff927f5319d9d206e782a73979ccb/packages/ai/src/generate-text/stream-text.ts#L686-L697
 */
export const isStreamContentPart = (part: LanguageModelV2StreamPart) => {
  return (
    part.type === 'text-delta' ||
    part.type === 'reasoning-delta' ||
    part.type === 'source' ||
    part.type === 'tool-call' ||
    part.type === 'tool-result' ||
    part.type === 'tool-input-start' ||
    part.type === 'tool-input-delta' ||
    part.type === 'raw'
  );
};

/**
 * Type guard to check if a value is a Retry object (has a model property with a MODEL)
 */
export const isRetry = <MODEL extends LanguageModelV2 | EmbeddingModelV2>(
  value: unknown,
): value is Retry<MODEL> =>
  isObject(value) && 'model' in value && isModelV2(value.model);
