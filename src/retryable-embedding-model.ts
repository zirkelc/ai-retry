import { delay } from '@ai-sdk/provider-utils';
import { BaseRetryableModel } from './base-retryable-model.js';
import { calculateExponentialBackoff } from './calculate-exponential-backoff.js';
import { countModelAttempts } from './count-model-attempts.js';
import { findRetryModel } from './find-retry-model.js';
import { prepareRetryError } from './prepare-retry-error.js';
import type {
  EmbeddingModel,
  EmbeddingModelCallOptions,
  EmbeddingModelEmbed,
  Retry,
  RetryContext,
  RetryErrorAttempt,
} from './types.js';
import { isAbortError } from './utils.js';

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
   * Get the retry call options overrides from a retry configuration.
   */
  private getRetryCallOptions(
    callOptions: EmbeddingModelCallOptions,
    currentRetry?: Retry<EmbeddingModel>,
  ): EmbeddingModelCallOptions {
    const retryOptions = currentRetry?.options ?? {};

    return {
      ...callOptions,
      values: retryOptions.values ?? callOptions.values,
      headers: retryOptions.headers ?? callOptions.headers,
      // Support deprecated providerOptions at top level for backward compatibility
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
  private async withRetry<RESULT extends EmbeddingModelEmbed>(input: {
    fn: (retryCallOptions: EmbeddingModelCallOptions) => Promise<RESULT>;
    callOptions: EmbeddingModelCallOptions;
    attempts?: Array<RetryErrorAttempt<EmbeddingModel>>;
  }): Promise<{
    result: RESULT;
    attempts: Array<RetryErrorAttempt<EmbeddingModel>>;
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
        // Don't retry if user manually aborted the request.
        // TimeoutError from AbortSignal.timeout() will still be handled by retry handlers.
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

    const { result } = await this.withRetry({
      fn: async (retryCallOptions) => {
        return this.currentModel.doEmbed(retryCallOptions);
      },
      callOptions: callOptions,
    });

    this.updateStickyModel(startModel);

    return result;
  }
}
