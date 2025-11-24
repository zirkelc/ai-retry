import type {
  LanguageModelV2,
  LanguageModelV2CallOptions,
  LanguageModelV2StreamPart,
} from '@ai-sdk/provider';
import { delay } from '@ai-sdk/provider-utils';
import { calculateExponentialBackoff } from './calculate-exponential-backoff.js';
import { countModelAttempts } from './count-model-attempts.js';
import { findRetryModel } from './find-retry-model.js';
import { prepareRetryError } from './prepare-retry-error.js';
import type {
  LanguageModelV2Generate,
  LanguageModelV2Stream,
  Retry,
  RetryAttempt,
  RetryableModelOptions,
  RetryContext,
  RetryErrorAttempt,
  RetryResultAttempt,
} from './types.js';
import { isGenerateResult, isStreamContentPart } from './utils.js';

export class RetryableLanguageModel implements LanguageModelV2 {
  readonly specificationVersion = 'v2';

  private baseModel: LanguageModelV2;
  private currentModel: LanguageModelV2;
  private options: RetryableModelOptions<LanguageModelV2>;

  get modelId() {
    return this.currentModel.modelId;
  }
  get provider() {
    return this.currentModel.provider;
  }

  get supportedUrls() {
    return this.currentModel.supportedUrls;
  }

  constructor(options: RetryableModelOptions<LanguageModelV2>) {
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
   * Execute a function with retry logic for handling errors
   */
  private async withRetry<
    RESULT extends LanguageModelV2Stream | LanguageModelV2Generate,
  >(input: {
    fn: (currentRetry?: Retry<LanguageModelV2>) => Promise<RESULT>;
    attempts?: Array<RetryAttempt<LanguageModelV2>>;
    abortSignal?: AbortSignal;
  }): Promise<{
    result: RESULT;
    attempts: Array<RetryAttempt<LanguageModelV2>>;
  }> {
    /**
     * Track all attempts.
     */
    const attempts: Array<RetryAttempt<LanguageModelV2>> = input.attempts ?? [];

    /**
     * Track current retry configuration.
     */
    let currentRetry: Retry<LanguageModelV2> | undefined;

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
        const currentAttempt: RetryAttempt<LanguageModelV2> = {
          ...previousAttempt,
          model: this.currentModel,
        };

        /**
         * Create a shallow copy of the attempts for testing purposes
         */
        const updatedAttempts = [...attempts];

        const context: RetryContext<LanguageModelV2> = {
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

        /**
         * Check if the result should trigger a retry (only for generate results, not streams)
         */
        if (isGenerateResult(result)) {
          const { retryModel, attempt } = await this.handleResult(
            result,
            attempts,
          );

          attempts.push(attempt);

          if (retryModel) {
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

            /**
             * Continue to the next iteration to retry
             */
            continue;
          }
        }

        return { result, attempts };
      } catch (error) {
        const { retryModel, attempt } = await this.handleError(error, attempts);

        attempts.push(attempt);

        if (retryModel.delay) {
          /**
           * Calculate exponential backoff delay based on the number of attempts for this specific model.
           * The delay grows exponentially: baseDelay * backoffFactor^attempts
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
   * Handle a successful result and determine if a retry is needed
   */
  private async handleResult(
    result: LanguageModelV2Generate,
    attempts: ReadonlyArray<RetryAttempt<LanguageModelV2>>,
  ) {
    const resultAttempt: RetryResultAttempt = {
      type: 'result',
      result: result,
      model: this.currentModel,
    };

    /**
     * Save the current attempt
     */
    const updatedAttempts = [...attempts, resultAttempt];

    const context: RetryContext<LanguageModelV2> = {
      current: resultAttempt,
      attempts: updatedAttempts,
    };

    const retryModel = await findRetryModel(this.options.retries, context);

    return { retryModel, attempt: resultAttempt };
  }

  /**
   * Handle an error and determine if a retry is needed
   */
  private async handleError(
    error: unknown,
    attempts: ReadonlyArray<RetryAttempt<LanguageModelV2>>,
  ) {
    const errorAttempt: RetryErrorAttempt<LanguageModelV2> = {
      type: 'error',
      error: error,
      model: this.currentModel,
    };

    /**
     * Save the current attempt
     */
    const updatedAttempts = [...attempts, errorAttempt];

    const context: RetryContext<LanguageModelV2> = {
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
    options: LanguageModelV2CallOptions,
  ): Promise<LanguageModelV2Generate> {
    /**
     * Always start with the original model
     */
    this.currentModel = this.baseModel;

    /**
     * If retries are disabled, bypass retry machinery entirely
     */
    if (this.isDisabled()) {
      return this.currentModel.doGenerate(options);
    }

    const { result } = await this.withRetry({
      fn: async (currentRetry) => {
        // Apply retry configuration if available
        const callOptions: LanguageModelV2CallOptions = {
          ...options,
          providerOptions:
            currentRetry?.providerOptions ?? options.providerOptions,
          abortSignal: currentRetry?.timeout
            ? AbortSignal.timeout(currentRetry.timeout)
            : options.abortSignal,
        };
        return this.currentModel.doGenerate(callOptions);
      },
      abortSignal: options.abortSignal,
    });

    return result;
  }

  async doStream(
    options: LanguageModelV2CallOptions,
  ): Promise<LanguageModelV2Stream> {
    /**
     * Always start with the original model
     */
    this.currentModel = this.baseModel;

    /**
     * If retries are disabled, bypass retry machinery entirely
     */
    if (this.isDisabled()) {
      return this.currentModel.doStream(options);
    }

    /**
     * Perform the initial call to doStream with retry logic to handle errors before any data is streamed.
     */
    let { result, attempts } = await this.withRetry({
      fn: async (currentRetry) => {
        // Apply retry configuration if available
        const callOptions: LanguageModelV2CallOptions = {
          ...options,
          providerOptions:
            currentRetry?.providerOptions ?? options.providerOptions,
          abortSignal: currentRetry?.timeout
            ? AbortSignal.timeout(currentRetry.timeout)
            : options.abortSignal,
        };
        return this.currentModel.doStream(callOptions);
      },
      abortSignal: options.abortSignal,
    });

    /**
     * Wrap the original stream to handle retries if an error occurs during streaming.
     */
    const retryableStream = new ReadableStream({
      start: async (controller) => {
        let reader:
          | ReadableStreamDefaultReader<LanguageModelV2StreamPart>
          | undefined;
        let isStreaming = false;

        while (true) {
          try {
            reader = result.stream.getReader();

            while (true) {
              const { done, value } = await reader.read();
              if (done) break;

              /**
               * If the stream part is an error and no data has been streamed yet, we can retry
               * Throw the error to trigger the retry logic in withRetry
               */
              if (value.type === 'error') {
                if (!isStreaming) {
                  // If no data has been streamed yet, we can retry
                  throw value.error;
                }
              }

              /**
               * Mark that streaming has started once we receive actual content
               */
              if (isStreamContentPart(value)) {
                isStreaming = true;
              }

              /**
               * Enqueue the chunk to the consumer of the stream
               */
              controller.enqueue(value);
            }

            controller.close();
            break;
          } catch (error) {
            /**
             * Check if the error from the stream can be retried.
             * Otherwise it will rethrow the error.
             */
            const { retryModel, attempt } = await this.handleError(
              error,
              attempts,
            );

            /**
             * Save the attempt
             */
            attempts.push(attempt);

            if (retryModel.delay) {
              /**
               * Calculate exponential backoff delay based on the number of attempts for this specific model.
               * The delay grows exponentially: baseDelay * backoffFactor^attempts
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
                abortSignal: options.abortSignal,
              });
            }

            this.currentModel = retryModel.model;

            /**
             * Retry the request by calling doStream again.
             * This will create a new stream.
             */
            const retriedResult = await this.withRetry({
              fn: async () => {
                const callOptions: LanguageModelV2CallOptions = {
                  ...options,
                  providerOptions:
                    retryModel.providerOptions ?? options.providerOptions,
                  abortSignal: retryModel.timeout
                    ? AbortSignal.timeout(retryModel.timeout)
                    : options.abortSignal,
                };
                return this.currentModel.doStream(callOptions);
              },
              attempts,
              abortSignal: options.abortSignal,
            });

            /**
             * Cancel the previous reader and stream if we are retrying
             */
            await reader?.cancel();

            result = retriedResult.result;
            attempts = retriedResult.attempts;
          } finally {
            reader?.releaseLock();
          }
        }
      },
    });

    return {
      ...result,
      stream: retryableStream,
    };
  }
}
