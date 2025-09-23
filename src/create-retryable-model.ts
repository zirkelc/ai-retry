import type {
  LanguageModelV2,
  LanguageModelV2CallOptions,
} from '@ai-sdk/provider';
import { getErrorMessage } from '@ai-sdk/provider-utils';
import { RetryError } from 'ai';
import { getModelKey } from './get-model-key.js';
import type {
  LanguageModelV2Generate,
  LanguageModelV2Stream,
} from './types.js';

/**
 * The context provided to Retryables with the current attempt and all previous attempts.
 */
export interface RetryContext<CURRENT extends RetryAttempt = RetryAttempt> {
  current: CURRENT;
  attempts: Array<RetryAttempt>;
  totalAttempts: number;
}

type RetryErrorAttempt = {
  type: 'error';
  error: unknown;
  model: LanguageModelV2;
};

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

  async doGenerate(
    options: LanguageModelV2CallOptions,
  ): Promise<LanguageModelV2Generate> {
    /**
     * Always start with the original model
     */
    this.currentModel = this.baseModel;

    /**
     * Track number of attempts
     */
    let totalAttempts = 0;

    /**
     * Track all attempts.
     */
    const attempts: Array<RetryAttempt> = [];

    /**
     * The previous attempt that triggered a retry, or undefined if this is the first attempt
     */
    let previousAttempt: RetryAttempt | undefined;

    while (true) {
      /**
       * Call the onRetry handler if provided.
       * Skip on the first attempt since no previous attempt exists yet.
       */
      if (previousAttempt) {
        /**
         * Current attempt context with the retry model
         */
        const currentAttempt: RetryAttempt = {
          ...previousAttempt,
          model: this.currentModel,
        };

        /**
         * Context for the onRetry handler
         */
        const context: RetryContext = {
          current: currentAttempt,
          attempts: attempts,
          totalAttempts,
        };

        /**
         * Call the onRetry handler if provided
         */
        this.options.onRetry?.(context);
      }

      totalAttempts++;

      try {
        const result = await this.currentModel.doGenerate(options);

        /**
         * Check if the result should trigger a retry
         */
        const resultAttempt: RetryResultAttempt = {
          type: 'result',
          result,
          model: this.currentModel,
        };

        /**
         * Add the current attempt to the list before checking for retries
         */
        attempts.push(resultAttempt);

        const resultContext: RetryContext = {
          current: resultAttempt,
          attempts: attempts,
          totalAttempts,
        };

        const nextModel = await this.findNextModel(resultContext);

        if (nextModel) {
          /**
           * Set the model for the next attempt
           */
          this.currentModel = nextModel;

          /**
           * Set the previous attempt that triggered this retry
           */
          previousAttempt = resultAttempt;

          /**
           * Continue to the next iteration to retry
           */
          continue;
        }

        /**
         * No retry needed, remove the attempt since it was successful and return the result
         */
        attempts.pop();
        return result;
      } catch (error) {
        /**
         * Current attempt with current error
         */
        const errorAttempt: RetryErrorAttempt = {
          type: 'error',
          error: error,
          model: this.currentModel,
        };

        /**
         * Save the current attempt
         */
        attempts.push(errorAttempt);

        /**
         * Context for the retryables and onError handler
         */
        const context: RetryContext<RetryErrorAttempt> = {
          current: errorAttempt,
          attempts: attempts,
          totalAttempts,
        };

        /**
         * Call the onError handler if provided
         */
        this.options.onError?.(context);

        const nextModel = await this.findNextModel(context);

        /**
         * Handler didn't return any models to try next, rethrow the error.
         * If we retried the request, wrap the error into a `RetryError` for better visibility.
         */
        if (!nextModel) {
          if (totalAttempts > 1) {
            const errorMessage = getErrorMessage(error);
            const errors = attempts.flatMap((a) =>
              isErrorAttempt(a)
                ? a.error
                : `Result with finishReason: ${a.result.finishReason}`,
            );

            throw new RetryError({
              message: `Failed after ${totalAttempts} attempts. Last error: ${errorMessage}`,
              reason: 'maxRetriesExceeded',
              errors,
            });
          }

          throw error;
        }

        /**
         * Set the model for the next attempt
         */
        this.currentModel = nextModel;

        /**
         * Set the previous attempt that triggered this retry
         */
        previousAttempt = errorAttempt;
      }
    }
  }

  async doStream(
    options: LanguageModelV2CallOptions,
  ): Promise<LanguageModelV2Stream> {
    throw new Error('Streaming not implemented');
  }
}

export function createRetryable(
  config: CreateRetryableOptions,
): LanguageModelV2 {
  return new RetryableModel(config);
}
