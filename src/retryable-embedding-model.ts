import { delay } from '@ai-sdk/provider-utils';
import { calculateExponentialBackoff } from './calculate-exponential-backoff.js';
import { countModelAttempts } from './count-model-attempts.js';
import { findRetryModel } from './find-retry-model.js';
import { prepareRetryError } from './prepare-retry-error.js';
import type {
  EmbeddingModel,
  EmbeddingModelCallOptions,
  EmbeddingModelEmbed,
  EmbeddingModelRetryCallOptions,
  Retry,
  RetryableModelOptions,
  RetryContext,
  RetryErrorAttempt,
} from './types.js';

export class RetryableEmbeddingModel<VALUE> implements EmbeddingModel<VALUE> {
  readonly specificationVersion = 'v2';

  private baseModel: EmbeddingModel<VALUE>;
  private currentModel: EmbeddingModel<VALUE>;
  private options: RetryableModelOptions<EmbeddingModel<VALUE>>;

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

  constructor(options: RetryableModelOptions<EmbeddingModel<VALUE>>) {
    this.options = options;
    this.baseModel = options.model;
    this.currentModel = options.model;
  }

  /**
   * Check if retries are disabled
   */
  private isDisabled(): boolean {
    if (this.options.disabled === undefined) {
      return false;
    }

    return typeof this.options.disabled === 'function'
      ? this.options.disabled()
      : this.options.disabled;
  }

  /**
   * Get the retry call options overrides from a retry configuration.
   */
  private getRetryCallOptions(
    callOptions: EmbeddingModelCallOptions<VALUE>,
    currentRetry?: Retry<EmbeddingModel<VALUE>>,
  ): EmbeddingModelCallOptions<VALUE> {
    const retryOptions = currentRetry?.options ?? {};

    return {
      ...callOptions,
      // Override values if specified (cast to VALUE[] since the type is generic)
      values: retryOptions.values ?? callOptions.values,
      // Override HTTP options
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
  private async withRetry<RESULT extends EmbeddingModelEmbed<VALUE>>(input: {
    fn: (currentRetry?: Retry<EmbeddingModel<VALUE>>) => Promise<RESULT>;
    callOptions: EmbeddingModelCallOptions<VALUE>;
    attempts?: Array<RetryErrorAttempt<EmbeddingModel<VALUE>>>;
    // abortSignal?: AbortSignal;
  }): Promise<{
    result: RESULT;
    attempts: Array<RetryErrorAttempt<EmbeddingModel<VALUE>>>;
  }> {
    /**
     * Track all attempts.
     */
    const attempts: Array<RetryErrorAttempt<EmbeddingModel<VALUE>>> =
      input.attempts ?? [];

    /**
     * Track current retry configuration.
     */
    let currentRetry: Retry<EmbeddingModel<VALUE>> | undefined;

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
        const currentAttempt: RetryErrorAttempt<EmbeddingModel<VALUE>> = {
          ...previousAttempt,
          model: this.currentModel,
        };

        /**
         * Create a shallow copy of the attempts for testing purposes
         */
        const updatedAttempts = [...attempts];

        const context: RetryContext<EmbeddingModel<VALUE>> = {
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
        const result = await input.fn(currentRetry);

        return { result, attempts };
      } catch (error) {
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
    attempts: ReadonlyArray<RetryErrorAttempt<EmbeddingModel<VALUE>>>,
    callOptions: EmbeddingModelRetryCallOptions<VALUE>,
  ) {
    const errorAttempt: RetryErrorAttempt<EmbeddingModel<VALUE>> = {
      type: 'error',
      error: error,
      model: this.currentModel,
      options: callOptions,
    };

    /**
     * Save the current attempt
     */
    const updatedAttempts = [...attempts, errorAttempt];

    const context: RetryContext<EmbeddingModel<VALUE>> = {
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
    callOptions: EmbeddingModelCallOptions<VALUE>,
  ): Promise<EmbeddingModelEmbed<VALUE>> {
    /**
     * Always start with the original model
     */
    this.currentModel = this.baseModel;

    /**
     * If retries are disabled, bypass retry machinery entirely
     */
    if (this.isDisabled()) {
      return this.currentModel.doEmbed(callOptions);
    }

    const { result } = await this.withRetry({
      fn: async (currentRetry) => {
        const retryCallOptions = this.getRetryCallOptions(
          callOptions,
          currentRetry,
        );

        return this.currentModel.doEmbed(retryCallOptions);
      },
      callOptions: callOptions,
    });

    return result;
  }
}
