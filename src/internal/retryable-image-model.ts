import { delay } from '@ai-sdk/provider-utils';
import { BaseRetryableModel } from './base-retryable-model.js';
import { evaluateError } from './evaluate-error.js';
import { mergeImageModelCallOptions } from './merge-retry-call-options.js';
import { resolveBackoffDelay } from './resolve-backoff-delay.js';
import { retryDiesOnAbortedSignal } from './retry-dies-on-aborted-signal.js';
import { createRetryTelemetry, type RetryTelemetry } from './telemetry.js';
import type {
  ImageModel,
  ImageModelCallOptions,
  ImageModelGenerate,
  OnRetryOverrides,
  Retry,
  RetryContext,
  RetryErrorAttempt,
} from '../types.js';
export class RetryableImageModel
  extends BaseRetryableModel<ImageModel>
  implements ImageModel
{
  readonly specificationVersion = 'v3';

  get modelId() {
    return this.currentModel.modelId;
  }

  get provider() {
    return this.currentModel.provider;
  }

  get maxImagesPerCall() {
    return this.currentModel.maxImagesPerCall;
  }

  /**
   * Execute a function with retry logic for handling errors
   */
  private async withRetry<RESULT extends ImageModelGenerate>(input: {
    fn: (retryCallOptions: ImageModelCallOptions) => Promise<RESULT>;
    callOptions: ImageModelCallOptions;
    attempts?: Array<RetryErrorAttempt<ImageModel>>;
    recorder?: RetryTelemetry;
  }): Promise<{
    result: RESULT;
    attempts: Array<RetryErrorAttempt<ImageModel>>;
    callOptions: ImageModelCallOptions;
  }> {
    /**
     * Track all attempts.
     */
    const attempts: Array<RetryErrorAttempt<ImageModel>> = input.attempts ?? [];

    /**
     * Track current retry configuration.
     */
    let currentRetry: Retry<ImageModel> | undefined;

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
      let onRetryOverrides: OnRetryOverrides<ImageModel> | undefined;
      if (previousAttempt) {
        const currentAttempt: RetryErrorAttempt<ImageModel> = {
          ...previousAttempt,
          model: this.currentModel,
        };

        /**
         * Create a shallow copy of the attempts for testing purposes
         */
        const updatedAttempts = [...attempts];

        const context: RetryContext<ImageModel> = {
          current: currentAttempt,
          attempts: updatedAttempts,
        };

        onRetryOverrides = (await this.options.onRetry?.(context)) ?? undefined;
      }

      /**
       * Get the retry call options overrides for this attempt
       */
      const retryCallOptions = mergeImageModelCallOptions({
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
        /**
         * `handleError` throws when no retry matched. Record the attempt as
         * failed before that error propagates.
         */
        let decision: Awaited<ReturnType<typeof this.handleError>>;
        try {
          decision = await this.handleError(error, attempts, retryCallOptions);
        } catch (finalError) {
          input.recorder?.endAttempt({
            attempt: attemptNumber,
            outcome: 'failure',
            error,
          });
          throw finalError;
        }

        const { retryModel, attempt } = decision;

        attempts.push(attempt);

        /**
         * If the inbound abort signal is already aborted and the chosen
         * retry does not supply a fresh deadline, the retry would die
         * instantly with the same abort. Rethrow rather than fire a
         * misleading retry against a dead signal.
         */
        if (
          retryDiesOnAbortedSignal(input.callOptions.abortSignal, retryModel)
        ) {
          input.recorder?.endAttempt({
            attempt: attemptNumber,
            outcome: 'failure',
            error,
          });
          throw error;
        }

        const calculatedDelay = resolveBackoffDelay(retryModel, attempts);

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
    attempts: ReadonlyArray<RetryErrorAttempt<ImageModel>>,
    callOptions: ImageModelCallOptions,
  ) {
    const { retryModel, attempt, finalError } = await evaluateError({
      error,
      model: this.currentModel,
      options: callOptions,
      attempts,
      retries: this.options.retries,
      onError: this.options.onError,
    });

    /**
     * No model to try next: surface the error (a `RetryError` wrapping all
     * attempts when more than one was made).
     */
    if (!retryModel) {
      throw finalError;
    }

    return { retryModel, attempt };
  }

  async doGenerate(
    callOptions: ImageModelCallOptions,
  ): Promise<ImageModelGenerate> {
    /**
     * Resolve the starting model (base or sticky)
     */
    const startModel = this.resolveStartModel();
    this.currentModel = startModel;

    /**
     * If retries are disabled, bypass retry machinery entirely
     */
    if (this.isDisabled()) {
      return this.currentModel.doGenerate(callOptions);
    }

    const recorder = await createRetryTelemetry(
      this.options.experimental_telemetry,
      {
        operation: 'doGenerate',
        genAiOperation: 'generate_content',
        provider: startModel.provider,
        modelId: startModel.modelId,
      },
    );

    let operationError: unknown;
    try {
      const {
        result,
        attempts,
        callOptions: finalCallOptions,
      } = await this.withRetry({
        fn: async (retryCallOptions) => {
          return this.currentModel.doGenerate(retryCallOptions);
        },
        callOptions: callOptions,
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
