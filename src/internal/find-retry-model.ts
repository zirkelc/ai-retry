import { getModelKey } from './get-model-key.js';
import { type GatewayResolver, resolveModel } from './resolve-model.js';
import type { CallRetries } from '../call/types.js';
import type {
  AnyModel,
  ResolvedModel,
  ModelRetries,
  Retry,
  ModelRetryable,
} from '../types.js';
import { isObject, isResultAttempt } from './guards.js';

/**
 * The parts of a retry context the shared retry internals actually read: which
 * kind of outcome the current attempt had, and which model each attempt ran
 * against.
 *
 * Written structurally so the model layer and the call layer — whose contexts
 * are deliberately unrelated types, and must stay that way — can both be driven
 * by one implementation.
 */
export type RetryContextLike = {
  current: { type: string; model: AnyModel };
  attempts: ReadonlyArray<{ model: AnyModel }>;
};

/**
 * The retry handlers, from either layer.
 *
 * A union rather than a single loosened shape, so that inference still works
 * from whichever of the two aliases the caller declared — and so a handler is
 * still checked against the layer whose list it was written into.
 */
export type RetriesLike<MODEL extends AnyModel, INPUT> =
  | ModelRetries<MODEL, INPUT>
  | CallRetries<MODEL, INPUT>;

/**
 * Find the next model to retry with based on the retry context.
 * `resolve` resolves gateway model-id strings for the caller's model
 * family (a bare string is ambiguous across families).
 */
export async function findRetryModel<MODEL extends AnyModel, INPUT>(
  retries: RetriesLike<MODEL, INPUT>,
  context: RetryContextLike,
  resolve?: GatewayResolver,
): Promise<Retry<ResolvedModel<MODEL>, INPUT> | undefined> {
  /**
   * Filter retryables based on attempt type:
   * - Result-based attempts: Only consider function retryables (skip plain models and static Retry objects)
   * - Error-based attempts: Consider all retryables (functions + plain models + static Retry objects)
   */
  const applicableRetries = isResultAttempt(context.current as any)
    ? retries.filter((retry) => typeof retry === 'function')
    : retries;

  /**
   * Iterate through the applicable retryables to find a model to retry with
   */
  for (const retry of applicableRetries) {
    let retryModel: Retry<MODEL, INPUT> | undefined;

    if (typeof retry === `function`) {
      /**
       * Function retryable - call it with context
       * The function can be either ModelRetryable<MODEL> or ModelRetryable<ResolvableLanguageModel>
       * At runtime, both work because the context is structurally compatible
       * We use type assertion here because TypeScript can't prove the union type compatibility
       */
      retryModel = await (retry as unknown as ModelRetryable<any, INPUT>)(
        context as never,
      );
    } else if (isObject(retry) && `model` in retry) {
      /** Static Retry object */
      retryModel = retry as unknown as Retry<MODEL, INPUT>;
    } else {
      /** Plain model */
      retryModel = { model: retry } as unknown as Retry<MODEL, INPUT>;
    }

    if (retryModel) {
      /**
       * The model can be string or an instance.
       * If it is a string, we need to resolve it to an instance.
       */
      const modelValue = retryModel.model;
      const resolvedModel = resolveModel(modelValue, resolve);

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
        } as Retry<ResolvedModel<MODEL>, INPUT>;
      }
    }
  }

  return undefined;
}
