import { delay } from '@ai-sdk/provider-utils';
import { BaseRetryableModel } from './base-retryable-model.js';
import { calculateExponentialBackoff } from './calculate-exponential-backoff.js';
import { countModelAttempts } from './count-model-attempts.js';
import { findRetryModel } from './find-retry-model.js';
import { prepareRetryError } from './prepare-retry-error.js';
import { mergeImageModelCallOptions } from './merge-retry-call-options.js';
import type {
  ImageModel,
  ImageModelCallOptions,
  ImageModelGenerate,
  OnRetryOverrides,
  Retry,
  RetryContext,
  RetryErrorAttempt,
} from './types.js';
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

      try {
        /**
         * Call the function that may need to be retried
         */
        const result = await input.fn(retryCallOptions);

        return { result, attempts, callOptions: retryCallOptions };
      } catch (error) {
        const { retryModel, attempt } = await this.handleError(
          error,
          attempts,
          retryCallOptions,
        );

        attempts.push(attempt);

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
          throw error;
        }

        if (retryModel.delay) {
          /**
           * Calculate exponential backoff delay based on the number of attempts for this specific model.
           * The delay grows exponentially: baseDelay * backoffFactor^attempts
           * Example: With delay=1000ms and backoffFactor=2:
           * - Attempt 1: 1000ms
           * - Attempt 2: 2000ms
           * - Attempt 3: 4000ms
           */
          const modelAttemptsCount = countModelAttempts(
            retryModel.model,
            attempts,
          );
          const calculatedDelay = calculateExponentialBackoff(
            retryModel.delay,
            retryModel.backoffFactor,
            modelAttemptsCount,
          );
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
    const errorAttempt: RetryErrorAttempt<ImageModel> = {
      type: 'error',
      error: error,
      model: this.currentModel,
      options: callOptions,
    };

    /**
     * Save the current attempt
     */
    const updatedAttempts = [...attempts, errorAttempt];

    const context: RetryContext<ImageModel> = {
      current: errorAttempt,
      attempts: updatedAttempts,
    };

    this.options.onError?.(context);

    const retryModel = await findRetryModel(this.options.retries, context);

    /**
     * Handler didn't return any models to try next, rethrow the error.
     * If we retried the request, wrap the error into a `RetryError` for better visibility.
     */
    if (!retryModel) {
      if (updatedAttempts.length > 1) {
        throw prepareRetryError(error, updatedAttempts);
      }

      throw error;
    }

    return { retryModel, attempt: errorAttempt };
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

    const {
      result,
      attempts,
      callOptions: finalCallOptions,
    } = await this.withRetry({
      fn: async (retryCallOptions) => {
        return this.currentModel.doGenerate(retryCallOptions);
      },
      callOptions: callOptions,
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
  }
}
