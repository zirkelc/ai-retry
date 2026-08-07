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
 * Any model the retry system accepts, already resolved to an instance.
 *
 * The bound almost every generic in the library carries. {@link AnyResolvableModel}
 * is the wider one, which also admits a gateway model-id string.
 */
export type AnyModel = LanguageModel | EmbeddingModel | ImageModel;

/**
 * Any model the retry system accepts, in resolvable (instance or gateway
 * string) form.
 */
export type AnyResolvableModel =
  | ResolvableLanguageModel
  | ResolvableEmbeddingModel
  | ResolvableImageModel;

export type ResolvableModel<MODEL extends AnyModel> =
  MODEL extends LanguageModel
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
 * The unified finish reason for a generation, as a provider reports it (nested
 * under `finishReason.unified`).
 *
 * The SDK entry points report the same set of values flat; a call-level
 * condition reads that one directly and has no need for this.
 */
export type ModelFinishReason = LanguageModelResult['finishReason']['unified'];

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
export type ModelRetryCallOptions<MODEL extends AnyModel> =
  MODEL extends LanguageModel
    ? LanguageModelRetryCallOptions
    : MODEL extends EmbeddingModel
      ? EmbeddingModelRetryCallOptions
      : ImageModelRetryCallOptions;

/**
 * Override returned by `onRetry` to influence the upcoming retry attempt.
 *
 * `INPUT` is the shape of the overridable call arguments. It defaults to the
 * provider-level call options, which is what the model wrappers use; the
 * call-level entry points substitute their own argument shape.
 */
export type OnRetryOverrides<
  MODEL extends AnyModel,
  INPUT = ModelRetryCallOptions<MODEL>,
> = { options?: INPUT };

/**
 * Maps a model type to its call options type.
 */
export type ModelCallOptions<MODEL extends AnyModel> =
  MODEL extends LanguageModel
    ? LanguageModelCallOptions
    : MODEL extends EmbeddingModel
      ? EmbeddingModelCallOptions
      : ImageModelCallOptions;

/**
 * Maps a model type to its result type.
 */
export type ModelResult<MODEL extends AnyModel> = MODEL extends LanguageModel
  ? LanguageModelResult | LanguageModelStream
  : MODEL extends EmbeddingModel
    ? EmbeddingModelEmbed
    : ImageModelGenerate;

/**
 * A retry attempt with an error
 */
export type ModelRetryErrorAttempt<MODEL extends AnyModel> = {
  type: 'error';
  error: unknown;
  result?: undefined;
  model: MODEL;
  /**
   * The call options used for this attempt.
   */
  options: ModelCallOptions<MODEL>;
};

/**
 * A retry attempt with a successful result
 */
export type ModelRetryResultAttempt = {
  type: 'result';
  /**
   * The generation result, provider-shaped — what the model returned.
   */
  result: LanguageModelResult;
  /**
   * The unified finish reason, lifted out of the provider's nested shape so a
   * condition need not dig for it.
   */
  finishReason: ModelFinishReason;
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
export type ModelRetryAttempt<MODEL extends AnyModel> =
  MODEL extends LanguageModel
    ? ModelRetryErrorAttempt<MODEL> | ModelRetryResultAttempt
    : ModelRetryErrorAttempt<MODEL>;

/**
 * The context provided to Retryables with the current attempt and all previous attempts.
 */
export type ModelRetryContext<MODEL extends AnyResolvableModel> = {
  /**
   * Current attempt that caused the retry
   */
  current: ModelRetryAttempt<ResolvedModel<MODEL>>;
  /**
   * All attempts made so far, including the current one
   */
  attempts: Array<ModelRetryAttempt<ResolvedModel<MODEL>>>;
};

/**
 * A successful attempt with the result
 */
export type ModelSuccessAttempt<MODEL extends AnyModel> = {
  type: 'success';
  model: MODEL;
  result: ModelResult<MODEL>;
  options: ModelCallOptions<MODEL>;
};

/**
 * The context provided to onSuccess with the successful attempt and all previous attempts.
 */
export type ModelSuccessContext<MODEL extends AnyResolvableModel> = {
  /**
   * The successful attempt
   */
  current: ModelSuccessAttempt<ResolvedModel<MODEL>>;
  /**
   * The preceding attempts that were retried, in order. Empty when the first
   * attempt succeeded. The successful attempt is `current` and is not repeated
   * here.
   */
  attempts: Array<ModelRetryAttempt<ResolvedModel<MODEL>>>;
};

/**
 * The context provided to onFailure when an operation terminally fails
 * (no retry matched, retries exhausted, or the retry itself failed).
 */
export type ModelFailureContext<MODEL extends AnyResolvableModel> = {
  /**
   * The final attempt that failed.
   */
  current: ModelRetryErrorAttempt<ResolvedModel<MODEL>>;
  /**
   * All attempts made, including the final failed one.
   */
  attempts: Array<ModelRetryAttempt<ResolvedModel<MODEL>>>;
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
   * Additional information to include in the telemetry data. Recorded on the
   * operation span as `ai_retry.metadata.<key>` attributes.
   */
  metadata?: Record<string, AttributeValue>;
}

/**
 * Options for creating a retryable model.
 */
export interface RetryableModelOptions<MODEL extends AnyModel> {
  model: MODEL;
  retries: ModelRetries<MODEL>;
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
  onError?: (context: ModelRetryContext<MODEL>) => void;
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
    context: ModelRetryContext<MODEL>,
  ) => void | OnRetryOverrides<MODEL> | Promise<void | OnRetryOverrides<MODEL>>;
  onSuccess?: (context: ModelSuccessContext<MODEL>) => void;
  /**
   * Called once when an operation terminally fails and the error could not
   * be recovered by a retry: no retry matched, all retries were exhausted,
   * or the retry itself failed. The counterpart to `onSuccess`.
   *
   * Not called when retries are disabled.
   */
  onFailure?: (context: ModelFailureContext<MODEL>) => void;
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
export type Retry<
  MODEL extends AnyResolvableModel,
  INPUT = ModelRetryCallOptions<ResolvedModel<MODEL>>,
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
   *
   * The shape is whatever the consuming API can override: written bare, the
   * provider-level call options the model wrappers accept; parameterized, the
   * entry point's own arguments for a call-level function.
   *
   * A retryable built by `.switch()`/`.retry()` leaves it unbound instead
   * (`never` when no options are given), which is what keeps one that sets no
   * options usable by both, and one that does set them checked against
   * whichever list it ends up in.
   */
  options?: INPUT;
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
export type ModelRetryable<
  MODEL extends AnyResolvableModel,
  INPUT = ModelRetryCallOptions<ResolvedModel<MODEL>>,
> = (
  context: ModelRetryContext<MODEL>,
) => Retry<MODEL, INPUT> | Promise<Retry<MODEL, INPUT> | undefined> | undefined;

/**
 * The configured retry handlers. `INPUT` is the shape `Retry.options` is
 * checked against; it defaults to the provider-level call options, which is
 * what the model wrappers accept.
 *
 * `Retry.options` sits in return position (a `ModelRetryable` returns a `Retry`),
 * so it is covariant: a retryable that sets no options has `options?: never`
 * and is accepted everywhere, while one that sets them only fits a target
 * whose `INPUT` covers them.
 */
export type ModelRetries<
  MODEL extends AnyModel,
  INPUT = ModelRetryCallOptions<MODEL>,
> = Array<
  | ModelRetryable<ResolvableModel<MODEL>, INPUT>
  | Retry<ResolvableModel<MODEL>, INPUT>
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

/* ------------------------------------------------------------------ *
 * Deprecated aliases
 *
 * These names predate the call-level retry API, when there was only one
 * layer and no prefix was needed. Each now has a `Model`-prefixed name that
 * says which layer it belongs to, alongside a `Call`-prefixed counterpart in
 * `src/call/types.ts`. The old names still work and are unchanged; they will
 * be removed when the model layer is.
 *
 * Not renamed, because they are genuinely shared by both layers: `Retry`,
 * `OnRetryOverrides`, `Reset`, `RetryTelemetrySettings`.
 * ------------------------------------------------------------------ */

/** @deprecated Use {@link ModelFinishReason}. */
export type FinishReason = ModelFinishReason;

/** @deprecated Use {@link ModelRetryCallOptions}. */
export type RetryCallOptions<MODEL extends AnyModel> =
  ModelRetryCallOptions<MODEL>;

/** @deprecated Use {@link ModelCallOptions}. */
export type CallOptions<MODEL extends AnyModel> = ModelCallOptions<MODEL>;

/** @deprecated Use {@link ModelResult}. */
export type Result<MODEL extends AnyModel> = ModelResult<MODEL>;

/** @deprecated Use {@link ModelRetryErrorAttempt}. */
export type RetryErrorAttempt<MODEL extends AnyModel> =
  ModelRetryErrorAttempt<MODEL>;

/** @deprecated Use {@link ModelRetryResultAttempt}. */
export type RetryResultAttempt = ModelRetryResultAttempt;

/** @deprecated Use {@link ModelRetryAttempt}. */
export type RetryAttempt<MODEL extends AnyModel> = ModelRetryAttempt<MODEL>;

/** @deprecated Use {@link ModelRetryContext}. */
export type RetryContext<MODEL extends AnyResolvableModel> =
  ModelRetryContext<MODEL>;

/** @deprecated Use {@link ModelSuccessAttempt}. */
export type SuccessAttempt<MODEL extends AnyModel> = ModelSuccessAttempt<MODEL>;

/** @deprecated Use {@link ModelSuccessContext}. */
export type SuccessContext<MODEL extends AnyResolvableModel> =
  ModelSuccessContext<MODEL>;

/** @deprecated Use {@link ModelFailureContext}. */
export type FailureContext<MODEL extends AnyResolvableModel> =
  ModelFailureContext<MODEL>;

/** @deprecated Use {@link ModelRetryable}. */
export type Retryable<
  MODEL extends AnyResolvableModel,
  INPUT = ModelRetryCallOptions<ResolvedModel<MODEL>>,
> = ModelRetryable<MODEL, INPUT>;

/** @deprecated Use {@link ModelRetries}. */
export type Retries<
  MODEL extends AnyModel,
  INPUT = ModelRetryCallOptions<MODEL>,
> = ModelRetries<MODEL, INPUT>;
