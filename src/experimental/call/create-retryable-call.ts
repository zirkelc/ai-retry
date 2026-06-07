import { delay } from '@ai-sdk/provider-utils';
import { BaseRetryableModel } from '../../internal/base-retryable-model.js';
import { evaluateError } from '../../internal/evaluate-error.js';
import { resolveAbortSignal } from '../../internal/merge-retry-call-options.js';
import { resolveBackoffDelay } from '../../internal/resolve-backoff-delay.js';
import { retryDiesOnAbortedSignal } from '../../internal/retry-dies-on-aborted-signal.js';
import { resolveModel } from '../../internal/resolve-model.js';
import { createRetryTelemetry } from '../../internal/telemetry.js';
import type {
  CallOptions,
  EmbeddingModel,
  ImageModel,
  LanguageModel,
  OnRetryOverrides,
  ProviderOptions,
  Reset,
  ResolvableModel,
  Retries,
  Retry,
  RetryableModelOptions,
  RetryAttempt,
  RetryCallOptions,
  RetryContext,
  RetryTelemetrySettings,
} from '../../types.js';

/**
 * Any model kind the driver can loop over. Defaults to `LanguageModel`, which
 * is the only kind with a call-level entry point today (`streamText`), so the
 * unparametrized aliases (`RetryCallAttempt`, `RetryableCallOptions`, …) stay
 * language-model-shaped for existing callers.
 */
type AnyModel = LanguageModel | EmbeddingModel | ImageModel;

/**
 * The per-attempt inputs handed to the call function. The retryable owns the
 * model selection and the per-attempt deadline so that each (re-)run gets a
 * fresh `abortSignal`; the call function only has to wire these into whatever
 * it invokes (e.g. an AI SDK `streamText`/`generateText` call).
 */
export type RetryCallAttempt<MODEL extends AnyModel = LanguageModel> = {
  /** Resolved model instance to use for this attempt. */
  model: MODEL;
  /** 1-based attempt number. */
  attempt: number;
  /**
   * Composed deadline for this attempt: the inbound caller signal merged with
   * a fresh `AbortSignal.timeout(...)` when a timeout applies. Pass this to the
   * underlying call rather than the caller's own signal so a re-run is not
   * killed instantly by an already-spent deadline.
   */
  abortSignal: AbortSignal | undefined;
  /**
   * Per-attempt call option overrides to apply on top of the call's own
   * options (from `Retry.options` and any `onRetry` return value).
   */
  options: RetryCallOptions<MODEL>;
};

/**
 * Options that influence a single `run` invocation.
 */
export type RetryCallRunOptions = {
  /** Genuine caller cancellation signal, composed into every attempt. */
  abortSignal?: AbortSignal;
  /**
   * Deadline in milliseconds for the first attempt. Subsequent attempts use
   * their matched `Retry.timeout`. Creating the deadline here (rather than
   * letting the caller bake it into the underlying call) is what lets a re-run
   * start from a fresh signal.
   */
  timeout?: number;
};

/**
 * The driver returned by {@link createRetryableCall}. Invoke it with a function
 * that performs one attempt; it loops over the configured retries until the
 * function returns (the result is passed through unchanged) or no retry
 * matches (the error is thrown, wrapped in a `RetryError` if more than one
 * attempt was made).
 */
export type RetryCall<MODEL extends AnyModel = LanguageModel> = <RESULT>(
  fn: (attempt: RetryCallAttempt<MODEL>) => Promise<RESULT>,
  runOptions?: RetryCallRunOptions,
) => Promise<RESULT>;

/**
 * Options for {@link createRetryableCall}.
 *
 * Mirrors the subset of `RetryableModelOptions` that applies to a generic
 * retry loop. There is no `onSuccess` here because the result is opaque to the
 * driver; the caller observes success directly from `run`'s return value.
 */
export interface RetryableCallOptions<MODEL extends AnyModel = LanguageModel> {
  /** Base model used for the first attempt (resolved on first use). */
  model: ResolvableModel<MODEL>;
  /** Retry handlers / fallback models, evaluated on each error. */
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
   * Telemetry configuration. When enabled, emits OpenTelemetry spans for retry
   * operations and attempts. Requires `@opentelemetry/api`.
   */
  experimental_telemetry?: RetryTelemetrySettings;
  onError?: (context: RetryContext<MODEL>) => void;
  /**
   * Called after a retry has been decided and the next model selected, but
   * before the retry call is issued. May return partial overrides for the
   * upcoming attempt.
   */
  onRetry?: (
    context: RetryContext<MODEL>,
  ) => void | OnRetryOverrides<MODEL> | Promise<void | OnRetryOverrides<MODEL>>;
}

/**
 * Resolve the per-attempt option overrides handed to the call function.
 *
 * Per-field precedence (highest → lowest):
 *   1. `onRetryOverrides.options.<field>`
 *   2. `currentRetry.options.<field>`
 *   3. `currentRetry.providerOptions` (deprecated top-level form, providerOptions only)
 */
function resolveRetryOptions<MODEL extends AnyModel>(
  currentRetry: Retry<MODEL> | undefined,
  onRetryOverrides: OnRetryOverrides<MODEL> | undefined,
): RetryCallOptions<MODEL> {
  const retryOptions = currentRetry?.options ?? {};
  const overrideOptions = onRetryOverrides?.options ?? {};
  const providerOptions =
    (overrideOptions as { providerOptions?: ProviderOptions })
      .providerOptions ??
    (retryOptions as { providerOptions?: ProviderOptions }).providerOptions ??
    currentRetry?.providerOptions;

  return {
    ...retryOptions,
    ...overrideOptions,
    ...(providerOptions ? { providerOptions } : {}),
  } as RetryCallOptions<MODEL>;
}

/**
 * Generic retry-loop driver. Unlike the model wrappers, this does not implement
 * a `LanguageModelV3`; it loops over an opaque async function and selects the
 * model + per-attempt deadline for each try. The call function decides what to
 * actually invoke, which keeps the driver independent of any specific AI SDK
 * entry point (`streamText`, `generateText`, …).
 *
 * Generic over the model kind; the resolved/conditional model types
 * (`ResolvedModel<MODEL>` from `findRetryModel`, `RetryAttempt<MODEL>`) collapse
 * to `MODEL` at runtime but TS can't prove it for a generic `MODEL`, so a few
 * casts bridge the gap — the same friction `evaluateError`/`findRetryModel`
 * already absorb internally.
 */
class RetryableCall<MODEL extends AnyModel> extends BaseRetryableModel<MODEL> {
  async run<RESULT>(
    fn: (attempt: RetryCallAttempt<MODEL>) => Promise<RESULT>,
    runOptions?: RetryCallRunOptions,
  ): Promise<RESULT> {
    /**
     * Resolve the starting model (base or sticky).
     */
    const startModel = this.resolveStartModel();
    this.currentModel = startModel;

    /**
     * If retries are disabled, bypass retry machinery entirely. The first
     * attempt still receives a composed deadline from the run options.
     */
    if (this.isDisabled()) {
      return fn({
        model: startModel,
        attempt: 1,
        abortSignal: resolveAbortSignal(
          runOptions?.abortSignal,
          runOptions?.timeout !== undefined
            ? ({ timeout: runOptions.timeout } as Retry<MODEL>)
            : undefined,
        ),
        options: {} as RetryCallOptions<MODEL>,
      });
    }

    const recorder = await createRetryTelemetry(
      this.options.experimental_telemetry,
      {
        operation: 'call',
        genAiOperation: 'chat',
        provider: startModel.provider,
        modelId: startModel.modelId,
      },
    );

    /**
     * Track all attempts. The driver is purely error-based: a returned result
     * is terminal and never re-evaluated.
     */
    const attempts: Array<RetryAttempt<MODEL>> = [];

    /**
     * Track current retry configuration.
     */
    let currentRetry: Retry<MODEL> | undefined;

    let operationError: unknown;
    try {
      while (true) {
        /**
         * Call the onRetry handler if provided. Skip on the first attempt
         * since no previous attempt exists yet.
         */
        let onRetryOverrides: OnRetryOverrides<MODEL> | undefined;
        const previousAttempt = attempts.at(-1);
        if (previousAttempt) {
          const currentAttempt = {
            ...previousAttempt,
            model: this.currentModel,
          };

          const context = {
            current: currentAttempt,
            attempts: [...attempts],
          } as unknown as RetryContext<MODEL>;

          onRetryOverrides =
            (await this.options.onRetry?.(context)) ?? undefined;
        }

        const attemptModel = this.currentModel;
        const attemptNumber = attempts.length + 1;

        /**
         * The deadline for this attempt: the matched retry's timeout, or the
         * first-attempt timeout from the run options.
         */
        const attemptTimeout = currentRetry?.timeout ?? runOptions?.timeout;
        const abortSignal = resolveAbortSignal(
          runOptions?.abortSignal,
          attemptTimeout !== undefined
            ? ({ timeout: attemptTimeout } as Retry<MODEL>)
            : currentRetry,
        );
        const options = resolveRetryOptions(currentRetry, onRetryOverrides);

        recorder?.startAttempt({
          attempt: attemptNumber,
          provider: attemptModel.provider,
          modelId: attemptModel.modelId,
          timeoutMs: attemptTimeout,
        });

        try {
          const result = await fn({
            model: attemptModel,
            attempt: attemptNumber,
            abortSignal,
            options,
          });

          recorder?.endAttempt({ attempt: attemptNumber, outcome: 'success' });
          this.updateStickyModel(startModel);
          return result;
        } catch (error) {
          /**
           * Evaluate the failure. `options` is a minimal placeholder: the
           * driver has no prompt of its own (the call function owns it), and
           * error retryables only read `error`/`model`.
           */
          const evaluation = await evaluateError({
            error,
            model: attemptModel,
            options: { abortSignal, ...options } as CallOptions<MODEL>,
            attempts,
            retries: this.options.retries,
            onError: this.options.onError,
          });

          const retryModel = evaluation.retryModel as Retry<MODEL> | undefined;
          const finalError = evaluation.finalError;

          attempts.push(evaluation.attempt as RetryAttempt<MODEL>);

          /**
           * No retry matched. Surface the error, wrapped in a `RetryError`
           * when more than one attempt was made.
           */
          if (!retryModel) {
            recorder?.endAttempt({
              attempt: attemptNumber,
              outcome: 'failure',
              error,
            });
            operationError = finalError;
            throw finalError;
          }

          /**
           * If the inbound caller signal is already aborted and the chosen
           * retry does not supply a fresh deadline, the retry would die
           * instantly with the same abort. Surface the error rather than fire
           * a misleading retry against a dead signal.
           */
          if (retryDiesOnAbortedSignal(runOptions?.abortSignal, retryModel)) {
            recorder?.endAttempt({
              attempt: attemptNumber,
              outcome: 'failure',
              error,
            });
            operationError = error;
            throw error;
          }

          const calculatedDelay = resolveBackoffDelay(retryModel, attempts);

          recorder?.endAttempt({
            attempt: attemptNumber,
            outcome: 'retry',
            error,
            delayMs: calculatedDelay,
          });

          if (calculatedDelay !== undefined) {
            await delay(calculatedDelay, {
              abortSignal: runOptions?.abortSignal,
            });
          }

          this.currentModel = retryModel.model;
          currentRetry = retryModel;
        }
      }
    } finally {
      recorder?.endOperation({
        provider: this.currentModel.provider,
        modelId: this.currentModel.modelId,
        error: operationError,
      });
    }
  }
}

/**
 * Create a generic, entry-point-agnostic retry-loop driver.
 *
 * The returned function loops over the configured `retries`, selecting the
 * model and a fresh per-attempt deadline for each try, and invoking the
 * supplied call function. Because the call function performs the actual work,
 * the driver stays independent of `streamText`/`generateText` and can wrap any
 * call whose deadline must be re-established on each retry.
 *
 * Defaults to `LanguageModel` — the only model kind with a call-level entry
 * point today — but is generic over the model kind for future call-level
 * retries (embeddings, images).
 */
export function createRetryableCall<MODEL extends AnyModel = LanguageModel>(
  options: RetryableCallOptions<MODEL>,
): RetryCall<MODEL> {
  const model = resolveModel(options.model);
  const instance = new RetryableCall<MODEL>({
    ...options,
    model,
  } as unknown as RetryableModelOptions<MODEL>);

  return (fn, runOptions) => instance.run(fn, runOptions);
}
