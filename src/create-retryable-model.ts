import type {
  LanguageModelV2,
  LanguageModelV2CallOptions,
  LanguageModelV2StreamPart,
} from '@ai-sdk/provider';
import { getErrorMessage } from '@ai-sdk/provider-utils';
import { RetryError } from 'ai';
import { getModelKey } from './get-model-key.js';
import type {
  LanguageModelV2Generate,
  LanguageModelV2Stream,
} from './types.js';
import { isGenerateResult, isStreamContentPart } from './utils.js';

/**
 * The context provided to Retryables with the current attempt and all previous attempts.
 */
export interface RetryContext<CURRENT extends RetryAttempt = RetryAttempt> {
  /**
   * Current attempt that caused the retry
   */
  current: CURRENT;
  /**
   * All attempts made so far, including the current one
   */
  attempts: Array<RetryAttempt>;
  /**
   * @deprecated Use `attempts.length` instead
   */
  totalAttempts: number;
}

/**
 * A retry attempt with an error
 */
type RetryErrorAttempt = {
  type: 'error';
  error: unknown;
  model: LanguageModelV2;
};

/**
 * A retry attempt with a successful result
 */
type RetryResultAttempt = {
  type: 'result';
  result: LanguageModelV2Generate;
  model: LanguageModelV2;
};

/**
 * A retry attempt with either an error or a result and the model used
 */
export type RetryAttempt = RetryErrorAttempt | RetryResultAttempt;

/**
 * Type guard to check if a retry attempt is an error attempt
 */
export function isErrorAttempt(
  attempt: RetryAttempt,
): attempt is RetryErrorAttempt {
  return attempt.type === 'error';
}

/**
 * Type guard to check if a retry attempt is a result attempt
 */
export function isResultAttempt(
  attempt: RetryAttempt,
): attempt is RetryResultAttempt {
  return attempt.type === 'result';
}

/**
 * A model to retry with and the maximum number of attempts for that model.
 */
export type RetryModel = {
  model: LanguageModelV2;
  maxAttempts?: number;
};

/**
 * A function that determines whether to retry with a different model based on the current attempt and all previous attempts.
 */
export type Retryable = (
  context: RetryContext,
) => RetryModel | Promise<RetryModel> | undefined;

/**
 * Options for creating a retryable model.
 */
export interface CreateRetryableOptions {
  model: LanguageModelV2;
  retries: Array<Retryable | LanguageModelV2>;
  onError?: (context: RetryContext<RetryErrorAttempt>) => void;
  onRetry?: (
    context: RetryContext<RetryErrorAttempt | RetryResultAttempt>,
  ) => void;
}

class RetryableModel implements LanguageModelV2 {
  readonly specificationVersion = 'v2';

  private baseModel: LanguageModelV2;
  private currentModel: LanguageModelV2;
  private options: CreateRetryableOptions;

  get modelId() {
    return this.currentModel.modelId;
  }
  get provider() {
    return this.currentModel.provider;
  }

  get supportedUrls() {
    return this.currentModel.supportedUrls;
  }

  constructor(options: CreateRetryableOptions) {
    this.options = options;
    this.baseModel = options.model;
    this.currentModel = options.model;
  }

  /**
   * Find the next model to retry with based on the retry context
   */
  private async findNextModel(
    context: RetryContext,
  ): Promise<LanguageModelV2 | undefined> {
    /**
     * Filter retryables based on attempt type:
     * - Result-based attempts: Only consider function retryables (skip plain models)
     * - Error-based attempts: Consider all retryables (functions + plain models)
     */
    const applicableRetries = isResultAttempt(context.current)
      ? this.options.retries.filter((retry) => typeof retry === 'function')
      : this.options.retries;

    /**
     * Iterate through the applicable retryables to find a model to retry with
     */
    for (const retry of applicableRetries) {
      const retryModel =
        typeof retry === 'function'
          ? await retry(context)
          : { model: retry, maxAttempts: 1 };

      if (retryModel) {
        /**
         * The model key uniquely identifies a model instance (provider + modelId)
         */
        const retryModelKey = getModelKey(retryModel.model);

        /**
         * Find all attempts with the same model
         */
        const retryAttempts = context.attempts.filter(
          (a) => getModelKey(a.model) === retryModelKey,
        );

        const maxAttempts = retryModel.maxAttempts ?? 1;

        /**
         * Check if the model can still be retried based on maxAttempts
         */
        if (retryAttempts.length < maxAttempts) {
          return retryModel.model;
        }
      }
    }

    return undefined;
  }

  /**
   * Execute a function with retry logic for handling errors
   */
  private async withRetry<
    RESULT extends LanguageModelV2Stream | LanguageModelV2Generate,
  >(input: {
    fn: () => Promise<RESULT>;
    attempts?: Array<RetryAttempt>;
  }): Promise<{
    result: RESULT;
    attempts: Array<RetryAttempt>;
  }> {
    /**
     * Track all attempts.
     */
    const attempts: Array<RetryAttempt> = input.attempts ?? [];

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
        const currentAttempt: RetryAttempt = {
          ...previousAttempt,
          model: this.currentModel,
        };

        /**
         * Create a shallow copy of the attempts for testing purposes
         */
        const updatedAttempts = [...attempts];

        const context: RetryContext = {
          current: currentAttempt,
          attempts: updatedAttempts,
          totalAttempts: updatedAttempts.length,
        };

        this.options.onRetry?.(context);
      }

      try {
        /**
         * Call the function that may need to be retried
         */
        const result = await input.fn();

        /**
         * Check if the result should trigger a retry (only for generate results, not streams)
         */
        if (isGenerateResult(result)) {
          const { nextModel, attempt } = await this.handleResult(
            result,
            attempts,
          );

          attempts.push(attempt);

          if (nextModel) {
            this.currentModel = nextModel;

            /**
             * Continue to the next iteration to retry
             */
            continue;
          }
        }

        return { result, attempts };
      } catch (error) {
        const { nextModel, attempt } = await this.handleError(error, attempts);

        attempts.push(attempt);

        this.currentModel = nextModel;
      }
    }
  }

  /**
   * Handle a successful result and determine if a retry is needed
   */
  private async handleResult(
    result: LanguageModelV2Generate,
    attempts: ReadonlyArray<RetryAttempt>,
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

    const resultContext: RetryContext = {
      current: resultAttempt,
      attempts: updatedAttempts,
      totalAttempts: updatedAttempts.length,
    };

    const nextModel = await this.findNextModel(resultContext);

    return { nextModel, attempt: resultAttempt };
  }

  /**
   * Handle an error and determine if a retry is needed
   */
  private async handleError(
    error: unknown,
    attempts: ReadonlyArray<RetryAttempt>,
  ) {
    const errorAttempt: RetryErrorAttempt = {
      type: 'error',
      error: error,
      model: this.currentModel,
    };

    /**
     * Save the current attempt
     */
    const updatedAttempts = [...attempts, errorAttempt];

    const context: RetryContext<RetryErrorAttempt> = {
      current: errorAttempt,
      attempts: updatedAttempts,
      totalAttempts: updatedAttempts.length,
    };

    this.options.onError?.(context);

    const nextModel = await this.findNextModel(context);

    /**
     * Handler didn't return any models to try next, rethrow the error.
     * If we retried the request, wrap the error into a `RetryError` for better visibility.
     */
    if (!nextModel) {
      if (updatedAttempts.length > 1) {
        throw this.prepareRetryError(error, updatedAttempts);
      }

      throw error;
    }

    return { nextModel, attempt: errorAttempt };
  }

  async doGenerate(
    options: LanguageModelV2CallOptions,
  ): Promise<LanguageModelV2Generate> {
    /**
     * Always start with the original model
     */
    this.currentModel = this.baseModel;

    const { result } = await this.withRetry({
      fn: async () => await this.currentModel.doGenerate(options),
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
     * Perform the initial call to doStream with retry logic to handle errors before any data is streamed.
     */
    let { result, attempts } = await this.withRetry({
      fn: async () => await this.currentModel.doStream(options),
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
            const { nextModel, attempt } = await this.handleError(
              error,
              attempts,
            );

            this.currentModel = nextModel;

            /**
             * Save the attempt
             */
            attempts.push(attempt);

            /**
             * Retry the request by calling doStream again.
             * This will create a new stream.
             */
            const retriedResult = await this.withRetry({
              fn: async () => await this.currentModel.doStream(options),
              attempts,
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

  private prepareRetryError(error: unknown, attempts: Array<RetryAttempt>) {
    const errorMessage = getErrorMessage(error);
    const errors = attempts.flatMap((a) =>
      isErrorAttempt(a)
        ? a.error
        : `Result with finishReason: ${a.result.finishReason}`,
    );

    return new RetryError(
      new RetryError({
        message: `Failed after ${attempts.length} attempts. Last error: ${errorMessage}`,
        reason: 'maxRetriesExceeded',
        errors,
      }),
    );
  }
}

export function createRetryable(
  config: CreateRetryableOptions,
): LanguageModelV2 {
  return new RetryableModel(config);
}
