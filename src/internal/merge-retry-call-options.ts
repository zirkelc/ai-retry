import type {
  EmbeddingModel,
  EmbeddingModelCallOptions,
  ImageModel,
  ImageModelCallOptions,
  LanguageModel,
  LanguageModelCallOptions,
  OnRetryOverrides,
  ProviderOptions,
  Retry,
} from '../types.js';
import { isTimeoutError } from './guards.js';

/**
 * Resolve `providerOptions` for the upcoming attempt.
 *
 * Precedence (highest → lowest):
 *   1. `onRetryOverrides.options.providerOptions`
 *   2. `currentRetry.options.providerOptions`
 *   3. `currentRetry.providerOptions` (deprecated top-level form)
 *   4. `base` (call options' own `providerOptions`)
 */
function resolveProviderOptions<
  MODEL extends LanguageModel | EmbeddingModel | ImageModel,
>(
  base: ProviderOptions,
  currentRetry: Retry<MODEL> | undefined,
  onRetryOverrides: OnRetryOverrides<MODEL> | undefined,
): ProviderOptions;
function resolveProviderOptions<
  MODEL extends LanguageModel | EmbeddingModel | ImageModel,
>(
  base: ProviderOptions | undefined,
  currentRetry: Retry<MODEL> | undefined,
  onRetryOverrides: OnRetryOverrides<MODEL> | undefined,
): ProviderOptions | undefined;
function resolveProviderOptions<
  MODEL extends LanguageModel | EmbeddingModel | ImageModel,
>(
  base: ProviderOptions | undefined,
  currentRetry: Retry<MODEL> | undefined,
  onRetryOverrides: OnRetryOverrides<MODEL> | undefined,
): ProviderOptions | undefined {
  return (
    onRetryOverrides?.options?.providerOptions ??
    currentRetry?.options?.providerOptions ??
    currentRetry?.providerOptions ??
    base
  );
}

/**
 * Resolve `abortSignal` for the upcoming attempt.
 *
 * If `currentRetry.timeout` is set, a fresh `AbortSignal.timeout(...)` is
 * created. When the base signal is still alive, the fresh deadline is
 * composed with the base so user cancellation still propagates mid-retry,
 * but a base `TimeoutError` (`AbortSignal.timeout` used as a wall-clock
 * budget) is ignored so it cannot truncate the retry's own deadline. When
 * the base is already aborted with a `TimeoutError`, it is dropped; with any
 * other reason it propagates. Without a retry timeout, the base is preserved
 * unchanged.
 */
function resolveAbortSignal<
  MODEL extends LanguageModel | EmbeddingModel | ImageModel,
>(
  base: AbortSignal | undefined,
  currentRetry: Retry<MODEL> | undefined,
): AbortSignal | undefined {
  if (currentRetry?.timeout === undefined) {
    return base;
  }

  const fresh = AbortSignal.timeout(currentRetry.timeout);
  if (base === undefined) {
    return fresh;
  }

  if (base.aborted) {
    return isTimeoutError(base.reason) ? fresh : base;
  }

  const controller = new AbortController();
  fresh.addEventListener('abort', () => controller.abort(fresh.reason), {
    once: true,
  });
  base.addEventListener(
    'abort',
    () => {
      if (isTimeoutError(base.reason)) return;
      controller.abort(base.reason);
    },
    { once: true },
  );
  return controller.signal;
}

/**
 * Merge call options for the upcoming language model retry attempt.
 *
 * Per-field precedence (highest → lowest):
 *   1. `onRetryOverrides.options.<field>`
 *   2. `currentRetry.options.<field>`
 *   3. `callOptions.<field>` (the base call options)
 */
export function mergeLanguageModelCallOptions(input: {
  callOptions: LanguageModelCallOptions;
  currentRetry?: Retry<LanguageModel>;
  onRetryOverrides?: OnRetryOverrides<LanguageModel>;
}): LanguageModelCallOptions {
  const { callOptions, currentRetry, onRetryOverrides } = input;
  const retryOptions = currentRetry?.options ?? {};
  const overrideOptions = onRetryOverrides?.options ?? {};

  return {
    ...callOptions,
    prompt: overrideOptions.prompt ?? retryOptions.prompt ?? callOptions.prompt,
    maxOutputTokens:
      overrideOptions.maxOutputTokens ??
      retryOptions.maxOutputTokens ??
      callOptions.maxOutputTokens,
    temperature:
      overrideOptions.temperature ??
      retryOptions.temperature ??
      callOptions.temperature,
    stopSequences:
      overrideOptions.stopSequences ??
      retryOptions.stopSequences ??
      callOptions.stopSequences,
    topP: overrideOptions.topP ?? retryOptions.topP ?? callOptions.topP,
    topK: overrideOptions.topK ?? retryOptions.topK ?? callOptions.topK,
    presencePenalty:
      overrideOptions.presencePenalty ??
      retryOptions.presencePenalty ??
      callOptions.presencePenalty,
    frequencyPenalty:
      overrideOptions.frequencyPenalty ??
      retryOptions.frequencyPenalty ??
      callOptions.frequencyPenalty,
    seed: overrideOptions.seed ?? retryOptions.seed ?? callOptions.seed,
    headers:
      overrideOptions.headers ?? retryOptions.headers ?? callOptions.headers,
    providerOptions: resolveProviderOptions<LanguageModel>(
      callOptions.providerOptions,
      currentRetry,
      onRetryOverrides,
    ),
    abortSignal: resolveAbortSignal<LanguageModel>(
      callOptions.abortSignal,
      currentRetry,
    ),
  };
}

/**
 * Merge call options for the upcoming embedding model retry attempt.
 */
export function mergeEmbeddingModelCallOptions(input: {
  callOptions: EmbeddingModelCallOptions;
  currentRetry?: Retry<EmbeddingModel>;
  onRetryOverrides?: OnRetryOverrides<EmbeddingModel>;
}): EmbeddingModelCallOptions {
  const { callOptions, currentRetry, onRetryOverrides } = input;
  const retryOptions = currentRetry?.options ?? {};
  const overrideOptions = onRetryOverrides?.options ?? {};

  return {
    ...callOptions,
    values: overrideOptions.values ?? retryOptions.values ?? callOptions.values,
    headers:
      overrideOptions.headers ?? retryOptions.headers ?? callOptions.headers,
    providerOptions: resolveProviderOptions<EmbeddingModel>(
      callOptions.providerOptions,
      currentRetry,
      onRetryOverrides,
    ),
    abortSignal: resolveAbortSignal<EmbeddingModel>(
      callOptions.abortSignal,
      currentRetry,
    ),
  };
}

/**
 * Merge call options for the upcoming image model retry attempt.
 */
export function mergeImageModelCallOptions(input: {
  callOptions: ImageModelCallOptions;
  currentRetry?: Retry<ImageModel>;
  onRetryOverrides?: OnRetryOverrides<ImageModel>;
}): ImageModelCallOptions {
  const { callOptions, currentRetry, onRetryOverrides } = input;
  const retryOptions = currentRetry?.options ?? {};
  const overrideOptions = onRetryOverrides?.options ?? {};

  return {
    ...callOptions,
    n: overrideOptions.n ?? retryOptions.n ?? callOptions.n,
    size: overrideOptions.size ?? retryOptions.size ?? callOptions.size,
    aspectRatio:
      overrideOptions.aspectRatio ??
      retryOptions.aspectRatio ??
      callOptions.aspectRatio,
    seed: overrideOptions.seed ?? retryOptions.seed ?? callOptions.seed,
    headers:
      overrideOptions.headers ?? retryOptions.headers ?? callOptions.headers,
    providerOptions: resolveProviderOptions<ImageModel>(
      callOptions.providerOptions,
      currentRetry,
      onRetryOverrides,
    ),
    abortSignal: resolveAbortSignal<ImageModel>(
      callOptions.abortSignal,
      currentRetry,
    ),
  };
}
