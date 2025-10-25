import type {
  EmbeddingModelV2 as AIEmbeddingModelV2,
  LanguageModelV2,
} from '@ai-sdk/provider';
import type { ProviderOptions } from '@ai-sdk/provider-utils';

export type EmbeddingModelV2<VALUE = any> = AIEmbeddingModelV2<VALUE>;

export type { LanguageModelV2 };

/**
 * Options for creating a retryable model.
 */
export interface RetryableModelOptions<
  MODEL extends LanguageModelV2 | EmbeddingModelV2,
> {
  model: MODEL;
  retries: Retries<MODEL>;
  onError?: (context: RetryContext<MODEL>) => void;
  onRetry?: (context: RetryContext<MODEL>) => void;
}

/**
 * The context provided to Retryables with the current attempt and all previous attempts.
 */
export type RetryContext<MODEL extends LanguageModelV2 | EmbeddingModelV2> = {
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
export type RetryErrorAttempt<
  MODEL extends LanguageModelV2 | EmbeddingModelV2,
> = {
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
  result: LanguageModelV2Generate;
  error?: undefined;
  model: LanguageModelV2;
};

/**
 * A retry attempt with either an error or a result and the model used
 */
export type RetryAttempt<MODEL extends LanguageModelV2 | EmbeddingModelV2> =
  | RetryErrorAttempt<MODEL>
  | RetryResultAttempt;

/**
 * A model to retry with and the maximum number of attempts for that model.
 */
export type Retry<MODEL extends LanguageModelV2 | EmbeddingModelV2> = {
  model: MODEL;
  maxAttempts?: number;
  delay?: number;
  backoffFactor?: number;
  providerOptions?: ProviderOptions;
};

/**
 * A function that determines whether to retry with a different model based on the current attempt and all previous attempts.
 */
export type Retryable<MODEL extends LanguageModelV2 | EmbeddingModelV2> = (
  context: RetryContext<MODEL>,
) => Retry<MODEL> | Promise<Retry<MODEL>> | undefined;

export type Retries<MODEL extends LanguageModelV2 | EmbeddingModelV2> = Array<
  Retryable<MODEL> | Retry<MODEL> | MODEL
>;

export type RetryableOptions<MODEL extends LanguageModelV2 | EmbeddingModelV2> =
  Partial<Omit<Retry<MODEL>, 'model'>>;

export type LanguageModelV2Generate = Awaited<
  ReturnType<LanguageModelV2['doGenerate']>
>;

export type LanguageModelV2Stream = Awaited<
  ReturnType<LanguageModelV2['doStream']>
>;

export type EmbeddingModelV2CallOptions<VALUE> = Parameters<
  EmbeddingModelV2<VALUE>['doEmbed']
>[0];

export type EmbeddingModelV2Embed<VALUE> = Awaited<
  ReturnType<EmbeddingModelV2<VALUE>['doEmbed']>
>;
