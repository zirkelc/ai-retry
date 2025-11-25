import { getModelKey } from './get-model-key.js';
import type {
  EmbeddingModel,
  LanguageModel,
  Retries,
  Retry,
  RetryContext,
} from './types.js';
import { isResultAttempt, isRetry } from './utils.js';

/**
 * Find the next model to retry with based on the retry context
 */
export async function findRetryModel<
  MODEL extends LanguageModel | EmbeddingModel<any>,
>(
  retries: Retries<MODEL>,
  context: RetryContext<MODEL>,
): Promise<Retry<MODEL> | undefined> {
  /**
   * Filter retryables based on attempt type:
   * - Result-based attempts: Only consider function retryables (skip plain models and static Retry objects)
   * - Error-based attempts: Consider all retryables (functions + plain models + static Retry objects)
   */
  const applicableRetries = isResultAttempt(context.current)
    ? retries.filter((retry) => typeof retry === 'function')
    : retries;

  /**
   * Iterate through the applicable retryables to find a model to retry with
   */
  for (const retry of applicableRetries) {
    let retryModel: Retry<MODEL> | undefined;

    if (typeof retry === 'function') {
      // Function retryable
      retryModel = await retry(context);
    } else if (isRetry(retry)) {
      // Static Retry object
      retryModel = retry;
    } else {
      // Plain model
      retryModel = { model: retry };
    }

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
        return retryModel;
      }
    }
  }

  return undefined;
}
