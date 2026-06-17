import type {
  EmbeddingModelV4,
  ImageModelV4,
  ImageModelV4CallOptions,
  LanguageModelV4,
  LanguageModelV4CallOptions,
  LanguageModelV4GenerateResult,
  LanguageModelV4Prompt,
  LanguageModelV4StreamPart,
  SharedV4ProviderOptions,
} from '@ai-sdk/provider';
import type { AttributeValue, Tracer } from '@opentelemetry/api';
import type { gateway } from 'ai';

type Literals<T> = T extends string
  ? string extends T
    ? never // It's `string` or `string & {}`, exclude it
    : T // It's a literal, keep it
  : never;

export type LanguageModel = LanguageModelV4;
export type EmbeddingModel = EmbeddingModelV4;
export type ImageModel = ImageModelV4;
export type LanguageModelCallOptions = LanguageModelV4CallOptions;
export type LanguageModelStreamPart = LanguageModelV4StreamPart;
export type ImageModelCallOptions = ImageModelV4CallOptions;
export type ProviderOptions = SharedV4ProviderOptions;

export type GatewayLanguageModelId = Parameters<
  (typeof gateway)['languageModel']
>[0];

export type GatewayEmbeddingModelId = Parameters<
  (typeof gateway)['embeddingModel']
>[0];

export type GatewayImageModelId = Parameters<(typeof gateway)['imageModel']>[0];

/**
 * A model that can be passed as either an instance or a gateway string
 * literal, resolved to an instance via the AI SDK Gateway.
 */
export type ResolvableLanguageModel =
  | LanguageModel
  | Literals<GatewayLanguageModelId>;
export type ResolvableEmbeddingModel =
  | EmbeddingModel
  | Literals<GatewayEmbeddingModelId>;
export type ResolvableImageModel = ImageModel | Literals<GatewayImageModelId>;

/**
 * Any model the retry system accepts, in resolvable (instance or gateway
 * string) form.
 */
export type AnyResolvableModel =
  | ResolvableLanguageModel
  | ResolvableEmbeddingModel
  | ResolvableImageModel;

export type ResolvableModel<
  MODEL extends LanguageModel | EmbeddingModel | ImageModel,
> = MODEL extends LanguageModel
  ? ResolvableLanguageModel
  : MODEL extends EmbeddingModel
    ? ResolvableEmbeddingModel
    : ResolvableImageModel;

export type ResolvedModel<MODEL extends AnyResolvableModel> =
  MODEL extends ResolvableLanguageModel
    ? LanguageModel
    : MODEL extends ResolvableEmbeddingModel
      ? EmbeddingModel
      : ImageModel;

/**
 * Result from a generateText call.
 */
export type LanguageModelResult = LanguageModelV4GenerateResult;

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
 * Maps a model type to its retry call options type — the subset of call
 * options that may be overridden for a single retry attempt.
 */
export type RetryCallOptions<
  MODEL extends LanguageModel | EmbeddingModel | ImageModel,
> = MODEL extends LanguageModel
  ? LanguageModelRetryCallOptions
  : MODEL extends EmbeddingModel
    ? EmbeddingModelRetryCallOptions
    : ImageModelRetryCallOptions;

/**
 * Override returned by `onRetry` to influence the upcoming retry attempt.
 */
export type OnRetryOverrides<
  MODEL extends LanguageModel | EmbeddingModel | ImageModel,
> = Pick<Retry<MODEL>, 'options'>;

/**
 * Maps a model type to its call options type.
 */
export type CallOptions<
  MODEL extends LanguageModel | EmbeddingModel | ImageModel,
> = MODEL extends LanguageModel
  ? LanguageModelCallOptions
  : MODEL extends EmbeddingModel
    ? EmbeddingModelCallOptions
    : ImageModelCallOptions;

/**
 * Maps a model type to its result type.
 */
export type Result<MODEL extends LanguageModel | EmbeddingModel | ImageModel> =
  MODEL extends LanguageModel
    ? LanguageModelResult | LanguageModelStream
    : MODEL extends EmbeddingModel
      ? EmbeddingModelEmbed
      : ImageModelGenerate;

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
  options: CallOptions<MODEL>;
};

/**
 * A retry attempt with a successful result
 */
export type RetryResultAttempt = {
  type: 'result';
  result: LanguageModelResult;
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
export type RetryContext<MODEL extends AnyResolvableModel> = {
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
 * A successful attempt with the result
 */
export type SuccessAttempt<
  MODEL extends LanguageModel | EmbeddingModel | ImageModel,
> = {
  type: 'success';
  model: MODEL;
  result: Result<MODEL>;
  options: CallOptions<MODEL>;
};

/**
 * The context provided to onSuccess with the successful attempt and all previous attempts.
 */
export type SuccessContext<MODEL extends AnyResolvableModel> = {
  /**
   * The successful attempt
   */
  current: SuccessAttempt<ResolvedModel<MODEL>>;
  /**
   * All attempts made so far, including the current one
   */
  attempts: Array<RetryAttempt<ResolvedModel<MODEL>>>;
};

/**
 * The context provided to onFailure when an operation terminally fails
 * (no retry matched, retries exhausted, or the retry itself failed).
 */
export type FailureContext<
  MODEL extends ResolvableLanguageModel | EmbeddingModel | ImageModel,
> = {
  /**
   * The final attempt that failed.
   */
  current: RetryErrorAttempt<ResolvedModel<MODEL>>;
  /**
   * All attempts made, including the final failed one.
   */
  attempts: Array<RetryAttempt<ResolvedModel<MODEL>>>;
  /**
   * The error surfaced to the caller. When more than one attempt was made,
   * this is a `RetryError` wrapping every attempt error; otherwise the raw
   * error.
   */
  error: unknown;
};

/**
 * Telemetry configuration for retry instrumentation.
 *
 * Talks to OpenTelemetry directly and independently of the AI SDK: when
 * enabled, each request emits a span for the operation with a child span per
 * attempt. Spans created here nest under any active span (e.g. the AI SDK's
 * `ai.generateText.doGenerate`, when that integration is registered) via
 * OpenTelemetry context propagation.
 *
 * The shape resembles the AI SDK's `telemetry` settings but is opt-in and
 * deliberately keeps a `tracer` field (which the AI SDK moved to its
 * `@ai-sdk/otel` integration), so retry spans work without adopting that
 * integration.
 *
 * Requires the optional peer dependency `@opentelemetry/api` to be installed
 * (in AI SDK v7 it is no longer a transitive dependency of `ai`; install
 * `@ai-sdk/otel` or `@opentelemetry/api` directly).
 */
export interface RetryTelemetrySettings {
  /**
   * Enable or disable retry telemetry. Disabled by default while experimental.
   */
  isEnabled?: boolean;
  /**
   * A custom tracer to use for the telemetry data. Defaults to the global
   * tracer (`trace.getTracer('ai-retry')`), which is a no-op until an
   * OpenTelemetry SDK is registered.
   */
  tracer?: Tracer;
  /**
   * Identifier for this function. Used to group telemetry data by function.
   */
  functionId?: string;
  /**
   * Additional information to include in the telemetry data. Recorded on the
   * operation span as `ai_retry.metadata.<key>` attributes.
   */
  metadata?: Record<string, AttributeValue>;
}

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

  /**
   * Experimental. Can change in patch versions without warning.
   *
   * Telemetry configuration. When enabled, emits OpenTelemetry spans for
   * retry operations and attempts. Requires `@opentelemetry/api`.
   */
  telemetry?: RetryTelemetrySettings;

  /**
   * @deprecated Use `telemetry` instead. Kept as an alias for compatibility;
   * when both are set, `telemetry` takes precedence.
   */
  experimental_telemetry?: RetryTelemetrySettings;

  // TODO: future iteration could let `onError` similarly decide whether a retry actually fires (today it is purely observational).
  onError?: (context: RetryContext<MODEL>) => void;
  /**
   * Called after a retry has been decided and the next model has been
   * selected, but before the retry call is issued.
   *
   * May optionally return a partial set of overrides for the upcoming
   * attempt.
   *
   * Precedence for the upcoming call:
   * base call options → `Retry.options` → `onRetry` return value (highest).
   *
   * Returning `undefined`/`void` leaves behavior unchanged.
   */
  onRetry?: (
    context: RetryContext<MODEL>,
  ) => void | OnRetryOverrides<MODEL> | Promise<void | OnRetryOverrides<MODEL>>;
  onSuccess?: (context: SuccessContext<MODEL>) => void;
  /**
   * Called once when an operation terminally fails and the error could not
   * be recovered by a retry: no retry matched, all retries were exhausted,
   * or the retry itself failed. The counterpart to `onSuccess`.
   *
   * Not called when retries are disabled.
   */
  onFailure?: (context: FailureContext<MODEL>) => void;
}

/**
 * A model to retry with and the maximum number of attempts for that model.
 *
 * The model can be:
 * - The exact MODEL type (instance)
 * - A gateway string literal (for any model family)
 * - A ResolvableModel<MODEL> (for compatibility with plain model arrays)
 *
 * This flexible approach allows retryable functions to return the exact model type
 * they received without type assertions, while still supporting string-based gateway models.
 */
export type Retry<MODEL extends AnyResolvableModel> = {
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
  options?: RetryCallOptions<ResolvedModel<MODEL>>;
  /**
   * @deprecated Use `options.providerOptions` instead.
   * Provider options to override for this retry.
   * If both `providerOptions` and `options.providerOptions` are set,
   * `options.providerOptions` takes precedence.
   */
  // TODO remove in this version
  providerOptions?: SharedV4ProviderOptions;
};

/**
 * A function that determines whether to retry with a different model based on the current attempt and all previous attempts.
 */
export type Retryable<MODEL extends AnyResolvableModel> = (
  context: RetryContext<MODEL>,
) => Retry<MODEL> | Promise<Retry<MODEL> | undefined> | undefined;

export type Retries<MODEL extends LanguageModel | EmbeddingModel | ImageModel> =
  Array<
    | Retryable<ResolvableModel<MODEL>>
    | Retry<ResolvableModel<MODEL>>
    | ResolvableModel<MODEL>
  >;

export type RetryableOptions<MODEL extends AnyResolvableModel> = Partial<
  Omit<Retry<MODEL>, 'model'>
>;

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
