import type {
  EmbeddingModelV3,
  ImageModelV3,
  ImageModelV3CallOptions,
  LanguageModelV3,
  LanguageModelV3CallOptions,
  LanguageModelV3Prompt,
  LanguageModelV3StreamPart,
  SharedV3ProviderOptions,
} from '@ai-sdk/provider';
import type { gateway } from 'ai';

type Literals<T> = T extends string
  ? string extends T
    ? never // It's `string` or `string & {}`, exclude it
    : T // It's a literal, keep it
  : never;

export type LanguageModel = LanguageModelV3;
export type EmbeddingModel = EmbeddingModelV3;
export type ImageModel = ImageModelV3;
export type LanguageModelCallOptions = LanguageModelV3CallOptions;
export type LanguageModelStreamPart = LanguageModelV3StreamPart;
export type ImageModelCallOptions = ImageModelV3CallOptions;
export type ProviderOptions = SharedV3ProviderOptions;

// export  type GatewayEmbeddingModelId = Parameters<typeof gateway['textEmbeddingModel']>[0];
export type GatewayLanguageModelId = Parameters<
  (typeof gateway)['languageModel']
>[0];

export type ResolvableLanguageModel =
  | LanguageModel
  | Literals<GatewayLanguageModelId>;

export type ResolvableModel<
  MODEL extends LanguageModel | EmbeddingModel | ImageModel,
> = MODEL extends LanguageModel
  ? ResolvableLanguageModel
  : MODEL extends EmbeddingModel
    ? EmbeddingModel
    : ImageModel;

export type ResolvedModel<
  MODEL extends ResolvableLanguageModel | EmbeddingModel | ImageModel,
> = MODEL extends ResolvableLanguageModel
  ? LanguageModel
  : MODEL extends EmbeddingModel
    ? EmbeddingModel
    : ImageModel;

/**
 * Call options that can be overridden during retry for language models.
 */
export type LanguageModelRetryCallOptions = Partial<
  Pick<
    LanguageModelCallOptions,
    | 'prompt'
    | 'maxOutputTokens'
    | 'temperature'
    | 'stopSequences'
    | 'topP'
    | 'topK'
    | 'presencePenalty'
    | 'frequencyPenalty'
    | 'seed'
    | 'headers'
    | 'providerOptions'
  >
>;

/**
 * Call options that can be overridden during retry for embedding models.
 */
export type EmbeddingModelRetryCallOptions = Partial<
  Pick<EmbeddingModelCallOptions, 'values' | 'headers' | 'providerOptions'>
>;

/**
 * Call options that can be overridden during retry for image models.
 */
export type ImageModelRetryCallOptions = Partial<
  Pick<
    ImageModelCallOptions,
    'n' | 'size' | 'aspectRatio' | 'seed' | 'headers' | 'providerOptions'
  >
>;

/**
 * A retry attempt with an error
 */
export type RetryErrorAttempt<
  MODEL extends LanguageModel | EmbeddingModel | ImageModel,
> = {
  type: 'error';
  error: unknown;
  result?: undefined;
  model: MODEL;
  /**
   * The call options used for this attempt.
   */
  options: MODEL extends LanguageModel
    ? LanguageModelCallOptions
    : MODEL extends EmbeddingModel
      ? EmbeddingModelCallOptions
      : ImageModelCallOptions;
};

/**
 * A retry attempt with a successful result
 */
export type RetryResultAttempt = {
  type: 'result';
  result: LanguageModelGenerate;
  error?: undefined;
  model: LanguageModel;
  /**
   * The call options used for this attempt.
   */
  options: LanguageModelCallOptions;
};

/**
 * A retry attempt with either an error or a result and the model used
 */
export type RetryAttempt<
  MODEL extends LanguageModel | EmbeddingModel | ImageModel,
> = MODEL extends LanguageModel
  ? RetryErrorAttempt<MODEL> | RetryResultAttempt
  : RetryErrorAttempt<MODEL>;

/**
 * The context provided to Retryables with the current attempt and all previous attempts.
 */
export type RetryContext<
  MODEL extends ResolvableLanguageModel | EmbeddingModel | ImageModel,
> = {
  /**
   * Current attempt that caused the retry
   */
  current: RetryAttempt<ResolvedModel<MODEL>>;
  /**
   * All attempts made so far, including the current one
   */
  attempts: Array<RetryAttempt<ResolvedModel<MODEL>>>;
};

/**
 * Options for creating a retryable model.
 */
export interface RetryableModelOptions<
  MODEL extends LanguageModel | EmbeddingModel | ImageModel,
> {
  model: MODEL;
  retries: Retries<MODEL>;
  disabled?: boolean | (() => boolean);
  /**
   * Controls when to reset back to the base model after a successful retry.
   *
   * @default 'after-request'
   */
  reset?: Reset;
  onError?: (context: RetryContext<MODEL>) => void;
  onRetry?: (context: RetryContext<MODEL>) => void;
}

/**
 * A model to retry with and the maximum number of attempts for that model.
 *
 * The model can be:
 * - The exact MODEL type (instance)
 * - A gateway string literal (for LanguageModel only)
 * - A ResolvableModel<MODEL> (for compatibility with plain model arrays)
 *
 * This flexible approach allows retryable functions to return the exact model type
 * they received without type assertions, while still supporting string-based gateway models.
 */
export type Retry<
  MODEL extends ResolvableLanguageModel | EmbeddingModel | ImageModel,
> = {
  model: MODEL;
  /**
   * Maximum number of attempts for this model.
   */
  maxAttempts?: number;
  /**
   * Delay in milliseconds before retrying.
   */
  delay?: number;
  /**
   * Factor to multiply the delay by for exponential backoff.
   */
  backoffFactor?: number;
  /**
   * Timeout in milliseconds for the retry request.
   * Creates a new AbortSignal with this timeout.
   */
  timeout?: number;
  /**
   * Call options to override for this retry.
   */
  options?: MODEL extends LanguageModel
    ? Partial<LanguageModelRetryCallOptions>
    : MODEL extends EmbeddingModel
      ? Partial<EmbeddingModelRetryCallOptions>
      : Partial<ImageModelRetryCallOptions>;
  /**
   * @deprecated Use `options.providerOptions` instead.
   * Provider options to override for this retry.
   * If both `providerOptions` and `options.providerOptions` are set,
   * `options.providerOptions` takes precedence.
   */
  providerOptions?: SharedV3ProviderOptions;
};

/**
 * A function that determines whether to retry with a different model based on the current attempt and all previous attempts.
 */
export type Retryable<
  MODEL extends ResolvableLanguageModel | EmbeddingModel | ImageModel,
> = (
  context: RetryContext<MODEL>,
) => Retry<MODEL> | Promise<Retry<MODEL> | undefined> | undefined;

export type Retries<MODEL extends LanguageModel | EmbeddingModel | ImageModel> =
  Array<
    | Retryable<ResolvableModel<MODEL>>
    | Retry<ResolvableModel<MODEL>>
    | ResolvableModel<MODEL>
  >;

export type RetryableOptions<
  MODEL extends ResolvableLanguageModel | EmbeddingModel | ImageModel,
> = Partial<Omit<Retry<MODEL>, 'model'>>;

/**
 * Controls when to reset the sticky model back to the base model.
 *
 * - `'after-request'` — reset after each request (default, current behavior)
 * - `` `after-${number}-requests` `` — use the retry model for the next N requests
 * - `` `after-${number}-seconds` `` — use the retry model for the next N seconds
 */
export type Reset =
  | 'after-request'
  | `after-${number}-requests`
  | `after-${number}-seconds`;

export type LanguageModelGenerate = Awaited<
  ReturnType<LanguageModel['doGenerate']>
>;

export type LanguageModelStream = Awaited<
  ReturnType<LanguageModel['doStream']>
>;

export type EmbeddingModelCallOptions = Parameters<
  EmbeddingModel['doEmbed']
>[0];

export type EmbeddingModelEmbed = Awaited<
  ReturnType<EmbeddingModel['doEmbed']>
>;

export type ImageModelGenerate = Awaited<ReturnType<ImageModel['doGenerate']>>;
