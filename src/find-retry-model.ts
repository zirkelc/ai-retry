import { getModelKey } from './get-model-key.js';
import type {
  EmbeddingModelV2,
  LanguageModelV2,
  Retries,
  RetryContext,
  RetryModel,
} from './types.js';
import { isResultAttempt } from './utils.js';

/**
 * Find the next model to retry with based on the retry context
 */
export async function findRetryModel<
  MODEL extends LanguageModelV2 | EmbeddingModelV2,
>(
  retries: Retries<MODEL>,
  context: RetryContext<MODEL>,
): Promise<RetryModel<MODEL> | undefined> {
  /**
   * Filter retryables based on attempt type:
   * - Result-based attempts: Only consider function retryables (skip plain models)
   * - Error-based attempts: Consider all retryables (functions + plain models)
   */
  const applicableRetries = isResultAttempt(context.current)
    ? retries.filter((retry) => typeof retry === 'function')
    : retries;

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
        return retryModel;
      }
    }
  }

  return undefined;
}
