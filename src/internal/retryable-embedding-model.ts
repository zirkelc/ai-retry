import { delay } from '@ai-sdk/provider-utils';
import { BaseRetryableModel } from './base-retryable-model.js';
import { calculateExponentialBackoff } from './calculate-exponential-backoff.js';
import { countModelAttempts } from './count-model-attempts.js';
import { findRetryModel } from './find-retry-model.js';
import { prepareRetryError } from './prepare-retry-error.js';
import { mergeEmbeddingModelCallOptions } from './merge-retry-call-options.js';
import { createRetryTelemetry, type RetryTelemetry } from './telemetry.js';
import type {
  EmbeddingModel,
  EmbeddingModelCallOptions,
  EmbeddingModelEmbed,
  OnRetryOverrides,
  Retry,
  RetryContext,
  RetryErrorAttempt,
} from '../types.js';
export class RetryableEmbeddingModel
  extends BaseRetryableModel<EmbeddingModel>
  implements EmbeddingModel
{
  readonly specificationVersion = 'v3';

  get modelId() {
    return this.currentModel.modelId;
  }

  get provider() {
    return this.currentModel.provider;
  }

  get maxEmbeddingsPerCall() {
    return this.currentModel.maxEmbeddingsPerCall;
  }

  get supportsParallelCalls() {
    return this.currentModel.supportsParallelCalls;
  }

  /**
   * Execute a function with retry logic for handling errors
   */
  private async withRetry<RESULT extends EmbeddingModelEmbed>(input: {
    fn: (retryCallOptions: EmbeddingModelCallOptions) => Promise<RESULT>;
    callOptions: EmbeddingModelCallOptions;
    attempts?: Array<RetryErrorAttempt<EmbeddingModel>>;
    recorder?: RetryTelemetry;
  }): Promise<{
    result: RESULT;
    attempts: Array<RetryErrorAttempt<EmbeddingModel>>;
    callOptions: EmbeddingModelCallOptions;
  }> {
    /**
     * Track all attempts.
     */
    const attempts: Array<RetryErrorAttempt<EmbeddingModel>> =
      input.attempts ?? [];

    /**
     * Track current retry configuration.
     */
    let currentRetry: Retry<EmbeddingModel> | undefined;

    while (true) {
      /**
       * The previous attempt that triggered a retry, or undefined if this is the first attempt
       */
      const previousAttempt = attempts.at(-1);

      /**
       * Call the onRetry handler if provided.
       * Skip on the first attempt since no previous attempt exists yet.
       */
      // TODO: future iteration could let `onError` similarly decide whether
      // a retry actually fires (today it is purely observational).
      let onRetryOverrides: OnRetryOverrides<EmbeddingModel> | undefined;
      if (previousAttempt) {
        const currentAttempt: RetryErrorAttempt<EmbeddingModel> = {
          ...previousAttempt,
          model: this.currentModel,
        };

        /**
         * Create a shallow copy of the attempts for testing purposes
         */
        const updatedAttempts = [...attempts];

        const context: RetryContext<EmbeddingModel> = {
          current: currentAttempt,
          attempts: updatedAttempts,
        };

        onRetryOverrides = (await this.options.onRetry?.(context)) ?? undefined;
      }

      /**
       * Get the retry call options overrides for this attempt
       */
      const retryCallOptions = mergeEmbeddingModelCallOptions({
        callOptions: input.callOptions,
        currentRetry,
        onRetryOverrides,
      });

      /**
       * The model and 1-based index for this attempt, captured for telemetry
       * before the call is issued.
       */
      const attemptModel = this.currentModel;
      const attemptNumber = attempts.length + 1;
      input.recorder?.startAttempt({
        attempt: attemptNumber,
        provider: attemptModel.provider,
        modelId: attemptModel.modelId,
        timeoutMs: currentRetry?.timeout,
      });

      try {
        /**
         * Call the function that may need to be retried
         */
        const result = await input.fn(retryCallOptions);

        input.recorder?.endAttempt({
          attempt: attemptNumber,
          outcome: 'success',
        });
        return { result, attempts, callOptions: retryCallOptions };
      } catch (error) {
        const { retryModel, attempt, finalError } = await this.handleError(
          error,
          attempts,
          retryCallOptions,
        );

        attempts.push(attempt);

        /**
         * No retry matched. Record the attempt as failed and throw the
         * surfaced error.
         */
        if (!retryModel) {
          input.recorder?.endAttempt({
            attempt: attemptNumber,
            outcome: 'failure',
            error,
          });
          throw finalError;
        }

        /**
         * If the inbound abort signal is already aborted and the chosen
         * retry does not supply a fresh deadline, the retry would die
         * instantly with the same abort. Rethrow rather than fire a
         * misleading retry against a dead signal.
         */
        if (
          input.callOptions.abortSignal?.aborted &&
          retryModel.timeout === undefined
        ) {
          input.recorder?.endAttempt({
            attempt: attemptNumber,
            outcome: 'failure',
            error,
          });
          throw error;
        }

        /**
         * Calculate exponential backoff delay based on the number of attempts
         * for this specific model: baseDelay * backoffFactor^attempts.
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

        input.recorder?.endAttempt({
          attempt: attemptNumber,
          outcome: 'retry',
          error,
          delayMs: calculatedDelay,
        });

        if (calculatedDelay !== undefined) {
          await delay(calculatedDelay, {
            abortSignal: retryCallOptions.abortSignal,
          });
        }

        this.currentModel = retryModel.model;
        currentRetry = retryModel;
      }
    }
  }

  /**
   * Handle an error and determine if a retry is needed
   */
  private async handleError(
    error: unknown,
    attempts: ReadonlyArray<RetryErrorAttempt<EmbeddingModel>>,
    callOptions: EmbeddingModelCallOptions,
  ) {
    const errorAttempt: RetryErrorAttempt<EmbeddingModel> = {
      type: 'error',
      error: error,
      model: this.currentModel,
      options: callOptions,
    };

    /**
     * Save the current attempt
     */
    const updatedAttempts = [...attempts, errorAttempt];

    const context: RetryContext<EmbeddingModel> = {
      current: errorAttempt,
      attempts: updatedAttempts,
    };

    this.options.onError?.(context);

    const retryModel = await findRetryModel(this.options.retries, context);

    /**
     * Handler didn't return any models to try next. Compute the error to
     * surface: if we retried the request, wrap it into a `RetryError` for
     * better visibility; otherwise surface the original error. The caller
     * pushes the attempt and throws `finalError`.
     */
    const finalError = retryModel
      ? undefined
      : updatedAttempts.length > 1
        ? prepareRetryError(error, updatedAttempts)
        : error;

    return { retryModel, attempt: errorAttempt, finalError };
  }

  /**
   * Fire the `onFailure` callback for a terminally failed operation. The
   * final attempt (last entry of `attempts`) is surfaced as `current`.
   */
  private emitFailure(
    attempts: Array<RetryErrorAttempt<EmbeddingModel>>,
    error: unknown,
  ) {
    if (!this.options.onFailure) return;
    const current = attempts.at(-1);
    if (!current) return;
    this.options.onFailure({ current, attempts, error });
  }

  async doEmbed(
    callOptions: EmbeddingModelCallOptions,
  ): Promise<EmbeddingModelEmbed> {
    /**
     * Resolve the starting model (base or sticky)
     */
    const startModel = this.resolveStartModel();
    this.currentModel = startModel;

    /**
     * If retries are disabled, bypass retry machinery entirely
     */
    if (this.isDisabled()) {
      return this.currentModel.doEmbed(callOptions);
    }

    const recorder = await createRetryTelemetry(
      this.options.experimental_telemetry,
      {
        operation: 'doEmbed',
        genAiOperation: 'embeddings',
        provider: startModel.provider,
        modelId: startModel.modelId,
      },
    );

    /**
     * Shared attempts array, threaded into `withRetry` so it stays populated
     * (including the final failed attempt) when the retry loop throws.
     */
    const attempts: Array<RetryErrorAttempt<EmbeddingModel>> = [];
    let operationError: unknown;
    try {
      const { result, callOptions: finalCallOptions } = await this.withRetry({
        fn: async (retryCallOptions) => {
          return this.currentModel.doEmbed(retryCallOptions);
        },
        callOptions: callOptions,
        attempts,
        recorder,
      });

      this.updateStickyModel(startModel);

      this.options.onSuccess?.({
        current: {
          type: 'success',
          model: this.currentModel,
          result,
          options: finalCallOptions,
        },
        attempts,
      });

      return result;
    } catch (error) {
      operationError = error;
      this.emitFailure(attempts, error);
      throw error;
    } finally {
      recorder?.endOperation({
        provider: this.currentModel.provider,
        modelId: this.currentModel.modelId,
        error: operationError,
      });
    }
  }
}
