import type {
  EmbeddingModel,
  LanguageModel,
  LanguageModelGenerate,
  LanguageModelStream,
  LanguageModelStreamPart,
  Retry,
  RetryAttempt,
  RetryErrorAttempt,
  RetryResultAttempt,
} from './types.js';

export const isObject = (value: unknown): value is Record<string, unknown> =>
  typeof value === 'object' && value !== null;

export const isString = (value: unknown): value is string =>
  typeof value === 'string';

export const isModel = (
  model: unknown,
): model is LanguageModel | EmbeddingModel =>
  isLanguageModel(model) || isEmbeddingModel(model);

export const isLanguageModel = (model: unknown): model is LanguageModel =>
  isObject(model) &&
  'provider' in model &&
  'modelId' in model &&
  'specificationVersion' in model &&
  model.specificationVersion === 'v3';

export const isEmbeddingModel = (model: unknown): model is EmbeddingModel =>
  isObject(model) &&
  'provider' in model &&
  'modelId' in model &&
  'specificationVersion' in model &&
  model.specificationVersion === 'v3';

export const isStreamResult = (
  result: LanguageModelGenerate | LanguageModelStream,
): result is LanguageModelStream => 'stream' in result;

export const isGenerateResult = (
  result: LanguageModelGenerate | LanguageModelStream,
): result is LanguageModelGenerate => 'content' in result;

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
export const isStreamContentPart = (part: LanguageModelStreamPart) => {
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
export const isRetry = <MODEL extends LanguageModel | EmbeddingModel>(
  value: unknown,
): value is Retry<MODEL> =>
  isObject(value) && 'model' in value && isModel(value.model);
