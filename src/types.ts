import type {
  EmbeddingModelV2,
  LanguageModelV2,
  LanguageModelV2CallOptions,
  LanguageModelV2Prompt,
  LanguageModelV2StreamPart,
  SharedV2ProviderOptions,
} from '@ai-sdk/provider';

export type LanguageModel = LanguageModelV2;
export type EmbeddingModel<VALUE = any> = EmbeddingModelV2<VALUE>;
export type LanguageModelCallOptions = LanguageModelV2CallOptions;
export type LanguageModelStreamPart = LanguageModelV2StreamPart;
export type LanguageModelPrompt = LanguageModelV2Prompt;

export type LanguageModelGenerate = Awaited<
  ReturnType<LanguageModel['doGenerate']>
>;

export type LanguageModelStream = Awaited<
  ReturnType<LanguageModel['doStream']>
>;

export type EmbeddingModelCallOptions<VALUE = any> = Parameters<
  EmbeddingModel<VALUE>['doEmbed']
>[0];

export type EmbeddingModelEmbed<VALUE = any> = Awaited<
  ReturnType<EmbeddingModel<VALUE>['doEmbed']>
>;

/**
 * Call options that can be overridden during retry for language models.
 * Excludes `abortSignal` (handled via `timeout`), `includeRawChunks` (internal),
 * `responseFormat`, `tools`, and `toolChoice`.
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
 * Excludes `abortSignal` (handled via `timeout`).
 */
export type EmbeddingModelRetryCallOptions<VALUE = any> = Partial<
  Pick<
    EmbeddingModelCallOptions<VALUE>,
    'values' | 'headers' | 'providerOptions'
  >
>;

/**
 * A retry attempt with an error
 */
export type RetryErrorAttempt<MODEL extends LanguageModel | EmbeddingModel> = {
  type: 'error';
  error: unknown;
  result?: undefined;
  model: MODEL;
  /**
   * The call options used for this attempt.
   */
  options: MODEL extends LanguageModel
    ? LanguageModelRetryCallOptions
    : EmbeddingModelRetryCallOptions;
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
  options: LanguageModelRetryCallOptions;
};

/**
 * A retry attempt with either an error or a result and the model used
 */
export type RetryAttempt<MODEL extends LanguageModel | EmbeddingModel> =
  | RetryErrorAttempt<MODEL>
  | RetryResultAttempt;

/**
 * The context provided to Retryables with the current attempt and all previous attempts.
 */
export type RetryContext<MODEL extends LanguageModel | EmbeddingModel> = {
  /**
   * Current attempt that caused the retry
   */
  current: RetryAttempt<MODEL>;
  /**
   * All attempts made so far, including the current one
   */
  attempts: Array<RetryAttempt<MODEL>>;
};

/**
 * Options for creating a retryable model.
 */
export interface RetryableModelOptions<
  MODEL extends LanguageModel | EmbeddingModel,
> {
  model: MODEL;
  retries: Retries<MODEL>;
  disabled?: boolean | (() => boolean);
  onError?: (context: RetryContext<MODEL>) => void;
  onRetry?: (context: RetryContext<MODEL>) => void;
}

/**
 * A model to retry with and the maximum number of attempts for that model.
 */
export type Retry<MODEL extends LanguageModel | EmbeddingModel> = {
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
    : Partial<EmbeddingModelRetryCallOptions>;
  /**
   * @deprecated Use `options.providerOptions` instead.
   * Provider options to override for this retry.
   * If both `providerOptions` and `options.providerOptions` are set,
   * `options.providerOptions` takes precedence.
   */
  providerOptions?: SharedV2ProviderOptions;
};

/**
 * A function that determines whether to retry with a different model based on the current attempt and all previous attempts.
 */
export type Retryable<MODEL extends LanguageModel | EmbeddingModel> = (
  context: RetryContext<MODEL>,
) => Retry<MODEL> | Promise<Retry<MODEL> | undefined> | undefined;

export type Retries<MODEL extends LanguageModel | EmbeddingModel> = Array<
  Retryable<MODEL> | Retry<MODEL> | MODEL
>;

export type RetryableOptions<MODEL extends LanguageModel | EmbeddingModel> =
  Partial<Omit<Retry<MODEL>, 'model'>>;
