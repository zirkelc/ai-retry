import { delay } from '@ai-sdk/provider-utils';
import { BaseRetryableModel } from '../../internal/base-retryable-model.js';
import { calculateExponentialBackoff } from '../../internal/calculate-exponential-backoff.js';
import { countModelAttempts } from '../../internal/count-model-attempts.js';
import { findRetryModel } from '../../internal/find-retry-model.js';
import { resolveAbortSignal } from '../../internal/merge-retry-call-options.js';
import { prepareRetryError } from '../../internal/prepare-retry-error.js';
import { resolveModel } from '../../internal/resolve-model.js';
import { createRetryTelemetry } from '../../internal/telemetry.js';
import type {
  LanguageModel,
  LanguageModelCallOptions,
  LanguageModelRetryCallOptions,
  OnRetryOverrides,
  Reset,
  ResolvableLanguageModel,
  Retries,
  Retry,
  RetryableModelOptions,
  RetryContext,
  RetryErrorAttempt,
  RetryTelemetrySettings,
} from '../../types.js';

/**
 * The per-attempt inputs handed to the call function. The retryable owns the
 * model selection and the per-attempt deadline so that each (re-)run gets a
 * fresh `abortSignal`; the call function only has to wire these into whatever
 * it invokes (e.g. an AI SDK `streamText`/`generateText` call).
 */
export type RetryCallAttempt = {
  /** Resolved model instance to use for this attempt. */
  model: LanguageModel;
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
  options: LanguageModelRetryCallOptions;
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
export type RetryCall = <RESULT>(
  fn: (attempt: RetryCallAttempt) => Promise<RESULT>,
  runOptions?: RetryCallRunOptions,
) => Promise<RESULT>;

/**
 * Options for {@link createRetryableCall}.
 *
 * Mirrors the subset of `RetryableModelOptions` that applies to a generic
 * retry loop. There is no `onSuccess` here because the result is opaque to the
 * driver; the caller observes success directly from `run`'s return value.
 */
export interface RetryableCallOptions {
  /** Base model used for the first attempt. */
  model: ResolvableLanguageModel;
  /** Retry handlers / fallback models, evaluated on each error. */
  retries: Retries<LanguageModel>;
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
  onError?: (context: RetryContext<LanguageModel>) => void;
  /**
   * Called after a retry has been decided and the next model selected, but
   * before the retry call is issued. May return partial overrides for the
   * upcoming attempt.
   */
  onRetry?: (
    context: RetryContext<LanguageModel>,
  ) =>
    | void
    | OnRetryOverrides<LanguageModel>
    | Promise<void | OnRetryOverrides<LanguageModel>>;
}

/**
 * Resolve the per-attempt option overrides handed to the call function.
 *
 * Per-field precedence (highest → lowest):
 *   1. `onRetryOverrides.options.<field>`
 *   2. `currentRetry.options.<field>`
 *   3. `currentRetry.providerOptions` (deprecated top-level form, providerOptions only)
 */
function resolveRetryOptions(
  currentRetry: Retry<LanguageModel> | undefined,
  onRetryOverrides: OnRetryOverrides<LanguageModel> | undefined,
): LanguageModelRetryCallOptions {
  const retryOptions = currentRetry?.options ?? {};
  const overrideOptions = onRetryOverrides?.options ?? {};
  const providerOptions =
    overrideOptions.providerOptions ??
    retryOptions.providerOptions ??
    currentRetry?.providerOptions;

  return {
    ...retryOptions,
    ...overrideOptions,
    ...(providerOptions ? { providerOptions } : {}),
  };
}

/**
 * Generic retry-loop driver. Unlike the model wrappers, this does not implement
 * a `LanguageModelV3`; it loops over an opaque async function and selects the
 * model + per-attempt deadline for each try. The call function decides what to
 * actually invoke, which keeps the driver independent of any specific AI SDK
 * entry point (`streamText`, `generateText`, …).
 */
class RetryableCall extends BaseRetryableModel<LanguageModel> {
  async run<RESULT>(
    fn: (attempt: RetryCallAttempt) => Promise<RESULT>,
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
            ? ({ timeout: runOptions.timeout } as Retry<LanguageModel>)
            : undefined,
        ),
        options: {},
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
     * Track all attempts. The driver only produces error attempts: a returned
     * result is terminal and never re-evaluated (the result is opaque here).
     */
    const attempts: Array<RetryErrorAttempt<LanguageModel>> = [];

    /**
     * Track current retry configuration.
     */
    let currentRetry: Retry<LanguageModel> | undefined;

    let operationError: unknown;
    try {
      while (true) {
        /**
         * Call the onRetry handler if provided. Skip on the first attempt
         * since no previous attempt exists yet.
         */
        let onRetryOverrides: OnRetryOverrides<LanguageModel> | undefined;
        const previousAttempt = attempts.at(-1);
        if (previousAttempt) {
          const currentAttempt: RetryErrorAttempt<LanguageModel> = {
            ...previousAttempt,
            model: this.currentModel,
          };

          const context: RetryContext<LanguageModel> = {
            current: currentAttempt,
            attempts: [...attempts],
          };

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
            ? ({ timeout: attemptTimeout } as Retry<LanguageModel>)
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
           * Build the error attempt. `options` is filled with a minimal valid
           * call-options object: the driver has no prompt of its own (the call
           * function owns it), and error retryables only read `error`/`model`.
           */
          const errorAttempt: RetryErrorAttempt<LanguageModel> = {
            type: 'error',
            error,
            model: attemptModel,
            options: {
              prompt: [],
              abortSignal,
              ...options,
            } as LanguageModelCallOptions,
          };

          const updatedAttempts = [...attempts, errorAttempt];

          const context: RetryContext<LanguageModel> = {
            current: errorAttempt,
            attempts: updatedAttempts,
          };

          this.options.onError?.(context);

          const retryModel = await findRetryModel(
            this.options.retries,
            context,
          );

          attempts.push(errorAttempt);

          /**
           * No retry matched. Surface the error, wrapped in a `RetryError`
           * when more than one attempt was made.
           */
          if (!retryModel) {
            const finalError =
              updatedAttempts.length > 1
                ? prepareRetryError(error, updatedAttempts)
                : error;
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
          if (
            runOptions?.abortSignal?.aborted &&
            retryModel.timeout === undefined
          ) {
            recorder?.endAttempt({
              attempt: attemptNumber,
              outcome: 'failure',
              error,
            });
            operationError = error;
            throw error;
          }

          /**
           * Calculate exponential backoff delay based on the number of
           * attempts for this specific model: baseDelay * backoffFactor^attempts.
           */
          let calculatedDelay: number | undefined;
          if (retryModel.delay) {
            const modelAttemptsCount = countModelAttempts(
              retryModel.model,
              attempts,
            );
            calculatedDelay = calculateExponentialBackoff(
              retryModel.delay,
              retryModel.backoffFactor,
              modelAttemptsCount,
            );
          }

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
 */
export function createRetryableCall(options: RetryableCallOptions): RetryCall {
  const model = resolveModel(options.model);
  const instance = new RetryableCall({
    ...options,
    model,
  } as unknown as RetryableModelOptions<LanguageModel>);

  return (fn, runOptions) => instance.run(fn, runOptions);
}
