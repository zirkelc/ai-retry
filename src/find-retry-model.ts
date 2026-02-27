import { getModelKey } from './get-model-key.js';
import { resolveModel } from './resolve-model.js';
import type {
  EmbeddingModel,
  ImageModel,
  LanguageModel,
  ResolvedModel,
  Retries,
  Retry,
  Retryable,
  RetryContext,
} from './types.js';
import { isObject, isResultAttempt } from './utils.js';

/**
 * Find the next model to retry with based on the retry context
 */
export async function findRetryModel<
  MODEL extends LanguageModel | EmbeddingModel | ImageModel,
>(
  retries: Retries<MODEL>,
  context: RetryContext<MODEL>,
): Promise<Retry<ResolvedModel<MODEL>> | undefined> {
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

    if (typeof retry === `function`) {
      /**
       * Function retryable - call it with context
       * The function can be either Retryable<MODEL> or Retryable<ResolvableLanguageModel>
       * At runtime, both work because the context is structurally compatible
       * We use type assertion here because TypeScript can't prove the union type compatibility
       */
      retryModel = await (retry as unknown as Retryable<MODEL>)(context);
    } else if (isObject(retry) && `model` in retry) {
      /** Static Retry object */
      retryModel = retry as unknown as Retry<MODEL>;
    } else {
      /** Plain model */
      retryModel = { model: retry } as unknown as Retry<MODEL>;
    }

    if (retryModel) {
      /**
       * The model can be string or an instance.
       * If it is a string, we need to resolve it to an instance.
       */
      const modelValue = retryModel.model;
      const resolvedModel = resolveModel(modelValue);

      /**
       * The model key uniquely identifies a model instance (provider + modelId)
       */
      const retryModelKey = getModelKey(resolvedModel);

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
        // Type assertion needed because TypeScript can't prove that
        // `MODEL extends LanguageModel` implies `ResolvedModel<MODEL> extends LanguageModel`
        // for the conditional `options` type, even though they are equivalent at runtime
        return {
          ...retryModel,
          model: resolvedModel,
        } as Retry<ResolvedModel<MODEL>>;
      }
    }
  }

  return undefined;
}
