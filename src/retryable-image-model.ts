import { delay } from '@ai-sdk/provider-utils';
import { BaseRetryableModel } from './base-retryable-model.js';
import { calculateExponentialBackoff } from './calculate-exponential-backoff.js';
import { countModelAttempts } from './count-model-attempts.js';
import { findRetryModel } from './find-retry-model.js';
import { prepareRetryError } from './prepare-retry-error.js';
import type {
  ImageModel,
  ImageModelCallOptions,
  ImageModelGenerate,
  Retry,
  RetryContext,
  RetryErrorAttempt,
} from './types.js';
import { isAbortError } from './utils.js';

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
   * Get the retry call options overrides from a retry configuration.
   */
  private getRetryCallOptions(
    callOptions: ImageModelCallOptions,
    currentRetry?: Retry<ImageModel>,
  ): ImageModelCallOptions {
    const retryOptions = currentRetry?.options ?? {};

    return {
      ...callOptions,
      n: retryOptions.n ?? callOptions.n,
      size: retryOptions.size ?? callOptions.size,
      aspectRatio: retryOptions.aspectRatio ?? callOptions.aspectRatio,
      seed: retryOptions.seed ?? callOptions.seed,
      headers: retryOptions.headers ?? callOptions.headers,
      /** Support deprecated providerOptions at top level for backward compatibility */
      providerOptions:
        retryOptions.providerOptions ??
        currentRetry?.providerOptions ??
        callOptions.providerOptions,
      abortSignal: currentRetry?.timeout
        ? AbortSignal.timeout(currentRetry.timeout)
        : callOptions.abortSignal,
    };
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

        this.options.onRetry?.(context);
      }

      /**
       * Get the retry call options overrides for this attempt
       */
      const retryCallOptions = this.getRetryCallOptions(
        input.callOptions,
        currentRetry,
      );

      try {
        /**
         * Call the function that may need to be retried
         */
        const result = await input.fn(retryCallOptions);

        return { result, attempts };
      } catch (error) {
        /** Don't retry if user manually aborted the request. */
        /** TimeoutError from AbortSignal.timeout() will still be handled by retry handlers. */
        if (isAbortError(error)) {
          throw error;
        }

        const { retryModel, attempt } = await this.handleError(
          error,
          attempts,
          retryCallOptions,
        );

        attempts.push(attempt);

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

    const { result } = await this.withRetry({
      fn: async (retryCallOptions) => {
        return this.currentModel.doGenerate(retryCallOptions);
      },
      callOptions: callOptions,
    });

    this.updateStickyModel(startModel);

    return result;
  }
}
