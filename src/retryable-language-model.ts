import { delay } from '@ai-sdk/provider-utils';
import { calculateExponentialBackoff } from './calculate-exponential-backoff.js';
import { countModelAttempts } from './count-model-attempts.js';
import { findRetryModel } from './find-retry-model.js';
import { prepareRetryError } from './prepare-retry-error.js';
import type {
  LanguageModel,
  LanguageModelCallOptions,
  LanguageModelGenerate,
  LanguageModelRetryCallOptions,
  LanguageModelStream,
  LanguageModelStreamPart,
  Retry,
  RetryAttempt,
  RetryableModelOptions,
  RetryContext,
  RetryErrorAttempt,
  RetryResultAttempt,
} from './types.js';
import { isGenerateResult, isStreamContentPart } from './utils.js';

export class RetryableLanguageModel implements LanguageModel {
  readonly specificationVersion = 'v3';

  private baseModel: LanguageModel;
  private currentModel: LanguageModel;
  private options: RetryableModelOptions<LanguageModel>;

  get modelId() {
    return this.currentModel.modelId;
  }
  get provider() {
    return this.currentModel.provider;
  }

  get supportedUrls() {
    return this.currentModel.supportedUrls;
  }

  constructor(options: RetryableModelOptions<LanguageModel>) {
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
    callOptions: LanguageModelCallOptions,
    currentRetry?: Retry<LanguageModel>,
  ): LanguageModelCallOptions {
    const retryOptions = currentRetry?.options ?? {};

    return {
      ...callOptions,
      prompt: retryOptions.prompt ?? callOptions.prompt,
      maxOutputTokens:
        retryOptions.maxOutputTokens ?? callOptions.maxOutputTokens,
      temperature: retryOptions.temperature ?? callOptions.temperature,
      stopSequences: retryOptions.stopSequences ?? callOptions.stopSequences,
      topP: retryOptions.topP ?? callOptions.topP,
      topK: retryOptions.topK ?? callOptions.topK,
      presencePenalty:
        retryOptions.presencePenalty ?? callOptions.presencePenalty,
      frequencyPenalty:
        retryOptions.frequencyPenalty ?? callOptions.frequencyPenalty,
      seed: retryOptions.seed ?? callOptions.seed,
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
  private async withRetry<
    RESULT extends LanguageModelStream | LanguageModelGenerate,
  >(input: {
    fn: (retryCallOptions: LanguageModelCallOptions) => Promise<RESULT>;
    callOptions: LanguageModelCallOptions;
    attempts?: Array<RetryAttempt<LanguageModel>>;
    currentRetry?: Retry<LanguageModel>;
  }): Promise<{
    result: RESULT;
    attempts: Array<RetryAttempt<LanguageModel>>;
  }> {
    /**
     * Track all attempts.
     */
    const attempts: Array<RetryAttempt<LanguageModel>> = input.attempts ?? [];

    /**
     * Track current retry configuration.
     */
    let currentRetry: Retry<LanguageModel> | undefined = input.currentRetry;

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
        const currentAttempt: RetryAttempt<LanguageModel> = {
          ...previousAttempt,
          model: this.currentModel,
        };

        /**
         * Create a shallow copy of the attempts for testing purposes
         */
        const updatedAttempts = [...attempts];

        const context: RetryContext<LanguageModel> = {
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

        /**
         * Check if the result should trigger a retry (only for generate results, not streams)
         */
        if (isGenerateResult(result)) {
          const { retryModel, attempt } = await this.handleResult(
            result,
            attempts,
            retryCallOptions,
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
              await delay(calculatedDelay, {
                abortSignal: retryCallOptions.abortSignal,
              });
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
   * Handle a successful result and determine if a retry is needed
   */
  private async handleResult(
    result: LanguageModelGenerate,
    attempts: ReadonlyArray<RetryAttempt<LanguageModel>>,
    callOptions: LanguageModelCallOptions,
  ) {
    const resultAttempt: RetryResultAttempt = {
      type: 'result',
      result: result,
      model: this.currentModel,
      options: callOptions,
    };

    /**
     * Save the current attempt
     */
    const updatedAttempts = [...attempts, resultAttempt];

    const context: RetryContext<LanguageModel> = {
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
    attempts: ReadonlyArray<RetryAttempt<LanguageModel>>,
    callOptions: LanguageModelCallOptions,
  ) {
    const errorAttempt: RetryErrorAttempt<LanguageModel> = {
      type: 'error',
      error: error,
      model: this.currentModel,
      options: callOptions,
    };

    /**
     * Save the current attempt
     */
    const updatedAttempts = [...attempts, errorAttempt];

    const context: RetryContext<LanguageModel> = {
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
    callOptions: LanguageModelCallOptions,
  ): Promise<LanguageModelGenerate> {
    /**
     * Always start with the original model
     */
    this.currentModel = this.baseModel;

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

    return result;
  }

  async doStream(
    callOptions: LanguageModelCallOptions,
  ): Promise<LanguageModelStream> {
    /**
     * Always start with the original model
     */
    this.currentModel = this.baseModel;

    /**
     * If retries are disabled, bypass retry machinery entirely
     */
    if (this.isDisabled()) {
      return this.currentModel.doStream(callOptions);
    }

    /**
     * Perform the initial call to doStream with retry logic to handle errors before any data is streamed.
     */
    let { result, attempts } = await this.withRetry({
      fn: async (retryCallOptions) => {
        return this.currentModel.doStream(retryCallOptions);
      },
      callOptions: callOptions,
    });

    /**
     * Track the current retry model for computing call options in the stream handler
     */
    let currentRetry: Retry<LanguageModel> | undefined;

    /**
     * Wrap the original stream to handle retries if an error occurs during streaming.
     */
    const retryableStream = new ReadableStream({
      start: async (controller) => {
        let reader:
          | ReadableStreamDefaultReader<LanguageModelStreamPart>
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
             * Get the retry call options for the failed attempt
             */
            const retryCallOptions = this.getRetryCallOptions(
              callOptions,
              currentRetry,
            );

            /**
             * Check if the error from the stream can be retried.
             * Otherwise it will rethrow the error.
             */
            const { retryModel, attempt } = await this.handleError(
              error,
              attempts,
              retryCallOptions,
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
                abortSignal: retryCallOptions.abortSignal,
              });
            }

            this.currentModel = retryModel.model;
            currentRetry = retryModel;

            /**
             * Retry the request by calling doStream again.
             * This will create a new stream.
             */
            const retriedResult = await this.withRetry({
              fn: async (retryCallOptions) => {
                return this.currentModel.doStream(retryCallOptions);
              },
              callOptions: callOptions,
              attempts,
              currentRetry,
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
