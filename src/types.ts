import type {
  EmbeddingModelV2,
  LanguageModelV2,
  LanguageModelV2CallOptions,
  LanguageModelV2Prompt,
  LanguageModelV2StreamPart,
  SharedV2ProviderOptions,
} from '@ai-sdk/provider';
import type { gateway } from 'ai';

type Literals<T> = T extends string
  ? string extends T
    ? never // It's `string` or `string & {}`, exclude it
    : T // It's a literal, keep it
  : never;

export type LanguageModel = LanguageModelV2;
export type EmbeddingModel<VALUE = any> = EmbeddingModelV2<VALUE>;
export type LanguageModelCallOptions = LanguageModelV2CallOptions;
export type LanguageModelStreamPart = LanguageModelV2StreamPart;
export type ProviderOptions = SharedV2ProviderOptions;

// export  type GatewayEmbeddingModelId = Parameters<typeof gateway['textEmbeddingModel']>[0];
export type GatewayLanguageModelId = Parameters<
  (typeof gateway)['languageModel']
>[0];

export type ResolvableLanguageModel =
  | LanguageModel
  | Literals<GatewayLanguageModelId>;

export type ResolvableModel<MODEL extends LanguageModel | EmbeddingModel> =
  MODEL extends LanguageModel ? ResolvableLanguageModel : EmbeddingModel;

export type ResolvedModel<
  MODEL extends ResolvableLanguageModel | EmbeddingModel,
> = MODEL extends ResolvableLanguageModel ? LanguageModel : EmbeddingModel;

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
    ? LanguageModelCallOptions
    : EmbeddingModelCallOptions<any>;
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
export type RetryAttempt<MODEL extends LanguageModel | EmbeddingModel> =
  | RetryErrorAttempt<MODEL>
  | RetryResultAttempt;

/**
 * The context provided to Retryables with the current attempt and all previous attempts.
 */
export type RetryContext<
  MODEL extends ResolvableLanguageModel | EmbeddingModel,
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
 *
 * The model can be:
 * - The exact MODEL type (instance)
 * - A gateway string literal (for LanguageModel only)
 * - A ResolvableModel<MODEL> (for compatibility with plain model arrays)
 *
 * This flexible approach allows retryable functions to return the exact model type
 * they received without type assertions, while still supporting string-based gateway models.
 */
export type Retry<MODEL extends ResolvableLanguageModel | EmbeddingModel> = {
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
export type Retryable<MODEL extends ResolvableLanguageModel | EmbeddingModel> =
  (
    context: RetryContext<MODEL>,
  ) => Retry<MODEL> | Promise<Retry<MODEL> | undefined> | undefined;

export type Retries<MODEL extends LanguageModel | EmbeddingModel> = Array<
  | Retryable<ResolvableModel<MODEL>>
  | Retry<ResolvableModel<MODEL>>
  | ResolvableModel<MODEL>
>;

export type RetryableOptions<
  MODEL extends ResolvableLanguageModel | EmbeddingModel,
> = Partial<Omit<Retry<MODEL>, 'model'>>;

export type LanguageModelGenerate = Awaited<
  ReturnType<LanguageModel['doGenerate']>
>;

export type LanguageModelStream = Awaited<
  ReturnType<LanguageModel['doStream']>
>;

export type EmbeddingModelCallOptions<VALUE> = Parameters<
  EmbeddingModel<VALUE>['doEmbed']
>[0];

export type EmbeddingModelEmbed<VALUE = any> = Awaited<
  ReturnType<EmbeddingModel<VALUE>['doEmbed']>
>;
