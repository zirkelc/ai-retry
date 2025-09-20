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

export interface RetryContext {
  error: unknown;
  baseModel: LanguageModelV2;
  currentModel: LanguageModelV2;
  triedModels: Map<string, RetryState>;
  totalAttempts: number;
}

export type RetryModel = {
  model: LanguageModelV2;
  maxAttempts: number;
};

export type Retryable = (
  context: RetryContext,
) => RetryModel | Promise<RetryModel> | undefined;

export type RetryState = {
  modelKey: string;
  model: LanguageModelV2;
  attempts: number;
  errors: Array<unknown>;
};

export interface CreateRetryableOptions {
  model: LanguageModelV2;
  retries: Array<Retryable | LanguageModelV2>;
  onError?: (context: RetryContext) => void;
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
     * Track models that have already been tried to avoid infinite loops
     */
    const triedModels = new Map<string, RetryState>();

    while (true) {
      totalAttempts++;

      try {
        return await this.currentModel.doGenerate(options);
      } catch (error) {
        const currentModelKey = getModelKey(this.currentModel);
        const prevState = triedModels.get(currentModelKey);

        /**
         * Save failed attempt with the current model
         */
        const newState: RetryState = {
          modelKey: currentModelKey,
          model: this.currentModel,
          attempts: (prevState?.attempts ?? 0) + 1,
          errors: [...(prevState?.errors ?? []), error],
        };

        triedModels.set(currentModelKey, newState);

        /**
         * Prepare context for the retry handlers
         */
        const context: RetryContext = {
          error,
          baseModel: this.baseModel,
          currentModel: this.currentModel,
          triedModels: triedModels,
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
            const retryModelKey = getModelKey(retryModel.model);
            const retryState = triedModels.get(retryModelKey);

            /**
             * Check if the model can still be retried based on maxAttempts
             */
            if (!retryState || retryState.attempts < retryModel.maxAttempts) {
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
            const newErrors = Array.from(triedModels.values()).flatMap(
              (state) => state.errors,
            );

            throw new RetryError({
              message: `Failed after ${totalAttempts} attempts. Last error: ${errorMessage}`,
              reason: 'maxRetriesExceeded',
              errors: newErrors,
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
