import { findRetryModel } from './find-retry-model.js';
import type { GatewayResolver } from './resolve-model.js';
import { prepareRetryError } from './prepare-retry-error.js';
import type {
  CallOptions,
  EmbeddingModel,
  ImageModel,
  LanguageModel,
  ResolvedModel,
  Retries,
  Retry,
  RetryAttempt,
  RetryContext,
  RetryErrorAttempt,
} from '../types.js';

/**
 * Evaluate a failed attempt against the configured retryables. Builds the error
 * attempt, notifies `onError`, and asks `findRetryModel` for the next model.
 *
 * Pure and model-agnostic — shared by the language/image/embedding model
 * wrappers and the generic call driver. The caller owns what happens next:
 * append `attempt` to its history, then either fail over to `retryModel` or, if
 * none matched, surface `finalError` (throw it, or enqueue it as a stream error
 * part). `finalError` is `undefined` when a retry matched, the original error on
 * the first attempt, and a `RetryError` wrapping all attempts thereafter.
 */
export async function evaluateError<
  MODEL extends LanguageModel | EmbeddingModel | ImageModel,
>(input: {
  error: unknown;
  model: MODEL;
  options: CallOptions<MODEL>;
  attempts: ReadonlyArray<RetryAttempt<MODEL>>;
  retries: Retries<MODEL>;
  onError?: (context: RetryContext<MODEL>) => void;
  /**
   * Resolves gateway model-id strings for the caller's model family. A bare
   * string is ambiguous across families, so each wrapper passes its own
   * resolver; defaults to the language-model resolver when omitted.
   */
  resolve?: GatewayResolver;
}): Promise<{
  retryModel: Retry<ResolvedModel<MODEL>> | undefined;
  attempt: RetryErrorAttempt<MODEL>;
  finalError: unknown;
}> {
  const errorAttempt: RetryErrorAttempt<MODEL> = {
    type: 'error',
    error: input.error,
    model: input.model,
    options: input.options,
  };

  /**
   * `ResolvedModel<MODEL>` collapses to `MODEL` for an already-resolved model,
   * but TS can't prove that for a generic `MODEL`, so the attempt list and
   * context need a cast — the same reason `findRetryModel` casts internally.
   */
  const updatedAttempts = [...input.attempts, errorAttempt] as Array<
    RetryAttempt<MODEL>
  >;

  const context = {
    current: errorAttempt,
    attempts: updatedAttempts,
  } as unknown as RetryContext<MODEL>;

  input.onError?.(context);

  const retryModel = await findRetryModel(input.retries, context, input.resolve);

  const finalError = retryModel
    ? undefined
    : updatedAttempts.length > 1
      ? prepareRetryError(input.error, updatedAttempts)
      : input.error;

  return { retryModel, attempt: errorAttempt, finalError };
}
