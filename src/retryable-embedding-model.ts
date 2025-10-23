import type { EmbeddingModelV2 } from '@ai-sdk/provider';
import { delay } from '@ai-sdk/provider-utils';
import { calculateExponentialBackoff } from './calculate-exponential-backoff.js';
import { countModelAttempts } from './count-model-attempts.js';
import { findRetryModel } from './find-retry-model.js';
import { prepareRetryError } from './prepare-retry-error.js';
import type {
  EmbeddingModelV2CallOptions,
  EmbeddingModelV2Embed,
  Retry,
  RetryableModelOptions,
  RetryContext,
  RetryErrorAttempt,
} from './types.js';

export class RetryableEmbeddingModel<VALUE> implements EmbeddingModelV2<VALUE> {
  readonly specificationVersion = 'v2';

  private baseModel: EmbeddingModelV2<VALUE>;
  private currentModel: EmbeddingModelV2<VALUE>;
  private options: RetryableModelOptions<EmbeddingModelV2<VALUE>>;

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

  constructor(options: RetryableModelOptions<EmbeddingModelV2<VALUE>>) {
    this.options = options;
    this.baseModel = options.model;
    this.currentModel = options.model;
  }

  /**
   * Execute a function with retry logic for handling errors
   */
  private async withRetry<RESULT extends EmbeddingModelV2Embed<VALUE>>(input: {
    fn: (currentRetry?: Retry<EmbeddingModelV2<VALUE>>) => Promise<RESULT>;
    attempts?: Array<RetryErrorAttempt<EmbeddingModelV2<VALUE>>>;
    abortSignal?: AbortSignal;
  }): Promise<{
    result: RESULT;
    attempts: Array<RetryErrorAttempt<EmbeddingModelV2<VALUE>>>;
  }> {
    /**
     * Track all attempts.
     */
    const attempts: Array<RetryErrorAttempt<EmbeddingModelV2<VALUE>>> =
      input.attempts ?? [];

    /**
     * Track current retry configuration.
     */
    let currentRetry: Retry<EmbeddingModelV2<VALUE>> | undefined;

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
        const currentAttempt: RetryErrorAttempt<EmbeddingModelV2<VALUE>> = {
          ...previousAttempt,
          model: this.currentModel,
        };

        /**
         * Create a shallow copy of the attempts for testing purposes
         */
        const updatedAttempts = [...attempts];

        const context: RetryContext<EmbeddingModelV2<VALUE>> = {
          current: currentAttempt,
          attempts: updatedAttempts,
        };

        this.options.onRetry?.(context);
      }

      try {
        /**
         * Call the function that may need to be retried
         */
        const result = await input.fn(currentRetry);

        return { result, attempts };
      } catch (error) {
        const { retryModel, attempt } = await this.handleError(error, attempts);

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
          await delay(calculatedDelay, { abortSignal: input.abortSignal });
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
    attempts: ReadonlyArray<RetryErrorAttempt<EmbeddingModelV2<VALUE>>>,
  ) {
    const errorAttempt: RetryErrorAttempt<EmbeddingModelV2<VALUE>> = {
      type: 'error',
      error: error,
      model: this.currentModel,
    };

    /**
     * Save the current attempt
     */
    const updatedAttempts = [...attempts, errorAttempt];

    const context: RetryContext<EmbeddingModelV2<VALUE>> = {
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
    options: EmbeddingModelV2CallOptions<VALUE>,
  ): Promise<EmbeddingModelV2Embed<VALUE>> {
    /**
     * Always start with the original model
     */
    this.currentModel = this.baseModel;

    const { result } = await this.withRetry({
      fn: async (currentRetry) => {
        // Apply retry configuration if available
        const callOptions = {
          ...options,
          providerOptions:
            currentRetry?.providerOptions ?? options.providerOptions,
        };
        return this.currentModel.doEmbed(callOptions);
      },
      abortSignal: options.abortSignal,
    });

    return result;
  }
}
