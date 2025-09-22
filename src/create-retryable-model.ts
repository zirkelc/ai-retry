import type {
  LanguageModelV2,
  LanguageModelV2CallOptions,
} from '@ai-sdk/provider';
import { getErrorMessage } from '@ai-sdk/provider-utils';
import { RetryError } from 'ai';
import { getModelKey } from './get-model-key.js';

type LanguageModelV2Generate = Awaited<
  ReturnType<LanguageModelV2['doGenerate']>
>;
type LanguageModelV2Stream = Awaited<ReturnType<LanguageModelV2['doStream']>>;

/**
 * The context provided to Retryables with the current attempt and all previous attempts.
 */
export interface RetryContext {
  current: RetryAttempt;
  attempts: Array<RetryAttempt>;
  totalAttempts: number;
}

/**
 * A retry attempt with the error and model used
 */
export interface RetryAttempt {
  error: unknown;
  model: LanguageModelV2;
}

/**
 * A model to retry with and the maximum number of attempts for that model.
 */
export type RetryModel = {
  model: LanguageModelV2;
  maxAttempts: number;
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
  onError?: (context: RetryContext) => void;
  onRetry?: (context: RetryContext) => void;
}

function defaultRetryModel(model: LanguageModelV2): RetryModel {
  return { model, maxAttempts: 1 };
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
     * The error occured in the previous attempt or undefined if this is the first attempt
     */
    let currentError: unknown | undefined;

    while (true) {
      /**
       * Call the onRetry handler if provided.
       * Skip on the first attempt since no error occured yet.
       */
      if (currentError) {
        /**
         * Current attempt with previous error
         */
        const currentAttempt: RetryAttempt = {
          error: currentError,
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

      try {
        totalAttempts++;
        return await this.currentModel.doGenerate(options);
      } catch (error) {
        /**
         * Save the error of the current attempt for the retry of the next iteration
         */
        currentError = error;

        /**
         * Current attempt with current error
         */
        const currentAttempt: RetryAttempt = {
          error: currentError,
          model: this.currentModel,
        };

        /**
         * Save the current attempt
         */
        attempts.push(currentAttempt);

        /**
         * Context for the retryables and onError handler
         */
        const context: RetryContext = {
          current: currentAttempt,
          attempts: attempts,
          totalAttempts,
        };

        /**
         * Call the onError handler if provided
         */
        this.options.onError?.(context);

        let nextModel: LanguageModelV2 | undefined;

        /**
         * Iterate through the retryables to find a model to retry with
         */
        for (const retry of this.options.retries) {
          const retryModel =
            typeof retry === 'function'
              ? await retry(context)
              : defaultRetryModel(retry);

          if (retryModel) {
            /**
             * The model key uniquely identifies a model instance (provider + modelId)
             */
            const retryModelKey = getModelKey(retryModel.model);

            /**
             * Find all attempts with the same model
             */
            const retryAttempts = attempts.filter(
              (a) => getModelKey(a.model) === retryModelKey,
            );

            /**
             * Check if the model can still be retried based on maxAttempts
             */
            if (retryAttempts.length < retryModel.maxAttempts) {
              nextModel = retryModel.model;
              break;
            }
          }
        }

        /**
         * Handler didn't return any models to try next, rethrow the error.
         * If we retried the request, wrap the error into a `RetryError` for better visibility.
         */
        if (!nextModel) {
          if (totalAttempts > 1) {
            const errorMessage = getErrorMessage(error);
            const errors = attempts.flatMap((a) => a.error);

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
