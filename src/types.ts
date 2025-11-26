import type {
  EmbeddingModelV3,
  LanguageModelV3,
  LanguageModelV3CallOptions,
  LanguageModelV3StreamPart,
} from '@ai-sdk/provider';
import type { ProviderOptions } from '@ai-sdk/provider-utils';

export type LanguageModel = LanguageModelV3;
export type EmbeddingModel<VALUE = unknown> = EmbeddingModelV3<VALUE>;
export type LanguageModelCallOptions = LanguageModelV3CallOptions;
export type LanguageModelStreamPart = LanguageModelV3StreamPart;

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
 * A retry attempt with an error
 */
export type RetryErrorAttempt<MODEL extends LanguageModel | EmbeddingModel> = {
  type: 'error';
  error: unknown;
  result?: undefined;
  model: MODEL;
};

/**
 * A retry attempt with a successful result
 */
export type RetryResultAttempt = {
  type: 'result';
  result: LanguageModelGenerate;
  error?: undefined;
  model: LanguageModel;
};

/**
 * A retry attempt with either an error or a result and the model used
 */
export type RetryAttempt<MODEL extends LanguageModel | EmbeddingModel> =
  | RetryErrorAttempt<MODEL>
  | RetryResultAttempt;

/**
 * A model to retry with and the maximum number of attempts for that model.
 */
export type Retry<MODEL extends LanguageModel | EmbeddingModel> = {
  model: MODEL;
  maxAttempts?: number;
  delay?: number;
  backoffFactor?: number;
  providerOptions?: ProviderOptions;
  timeout?: number;
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
