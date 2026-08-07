import { findRetryModel, type RetriesLike } from './find-retry-model.js';
import type { GatewayResolver } from './resolve-model.js';
import { prepareRetryError } from './prepare-retry-error.js';
import type { AnyModel, ResolvedModel, Retry } from '../types.js';
import type { RetryContextLike } from './find-retry-model.js';

/**
 * A failed attempt, with the shape of its call arguments left open.
 *
 * The two layers record different arguments — provider call options below a
 * model, the entry point's own arguments around a call — and nothing here reads
 * them, so the difference is simply carried through.
 */
export type ErrorAttemptLike<MODEL, OPTIONS> = {
  type: 'error';
  error: unknown;
  result?: undefined;
  model: MODEL;
  options: OPTIONS;
};

/**
 * Evaluate a failed attempt against the configured retryables. Builds the error
 * attempt, notifies `onError`, and asks `findRetryModel` for the next model.
 *
 * Pure and model-agnostic — shared by the language/image/embedding model
 * wrappers and the call-level retry loop. The caller owns what happens next:
 * append `attempt` to its history, then either fail over to `retryModel` or, if
 * none matched, surface `finalError` (throw it, or enqueue it as a stream error
 * part). `finalError` is `undefined` when a retry matched, the original error on
 * the first attempt, and a `RetryError` wrapping all attempts thereafter.
 */
export async function evaluateError<
  MODEL extends AnyModel,
  INPUT,
  OPTIONS,
>(input: {
  error: unknown;
  model: MODEL;
  options: OPTIONS;
  /**
   * The attempts made before this one. Only each attempt's model is read, so
   * either layer's attempt type serves.
   */
  attempts: ReadonlyArray<unknown>;
  retries: RetriesLike<MODEL, INPUT>;
  /**
   * Called with the layer's own context. Left open here for the same reason the
   * retryables are: which context is built is the caller's business.
   */
  onError?: (context: never) => void;
  /**
   * Resolves gateway model-id strings for the caller's model family. A bare
   * string is ambiguous across families, so each wrapper passes its own
   * resolver; defaults to the language-model resolver when omitted.
   */
  resolve?: GatewayResolver;
}): Promise<{
  retryModel: Retry<ResolvedModel<MODEL>, INPUT> | undefined;
  attempt: ErrorAttemptLike<MODEL, OPTIONS>;
  finalError: unknown;
}> {
  const errorAttempt: ErrorAttemptLike<MODEL, OPTIONS> = {
    type: 'error',
    error: input.error,
    model: input.model,
    options: input.options,
  };

  const updatedAttempts = [...input.attempts, errorAttempt];

  const context = {
    current: errorAttempt,
    attempts: updatedAttempts,
  } as unknown as RetryContextLike;

  input.onError?.(context as never);

  const retryModel = await findRetryModel<MODEL, INPUT>(
    input.retries,
    context,
    input.resolve,
  );

  const finalError = retryModel
    ? undefined
    : updatedAttempts.length > 1
      ? prepareRetryError(
          input.error,
          updatedAttempts as ReadonlyArray<{ type: string }>,
        )
      : input.error;

  return { retryModel, attempt: errorAttempt, finalError };
}
