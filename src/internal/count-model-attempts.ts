import { RetryError } from 'ai';
import { AiRetryError } from './ai-retry-error.js';
import { getModelKey } from './get-model-key.js';
import type { CallRetryAttempt } from '../call/types.js';
import type { AnyModel, ModelRetryAttempt } from '../types.js';

/**
 * The parts of an attempt of either layer read here.
 */
type AttemptLike = { model: AnyModel; error?: unknown };

/**
 * The attempts a retryable model made inside one attempt, when the attempt's
 * error comes from one.
 *
 * Usually the error arrives as the retryable model threw it. When the SDK's own
 * `maxRetries` ran the retryable model more than once, the SDK wraps each run's
 * error in a `RetryError` of its own, so its `errors` are read one level down,
 * each as a run on `model`. An error that names a retryable model's failure as
 * its `cause` counts as that failure.
 */
function innerAttempts(
  error: unknown,
  model: AnyModel,
): ReadonlyArray<AttemptLike> | undefined {
  if (AiRetryError.isInstance(error)) return error.attempts;

  if (
    RetryError.isInstance(error) &&
    error.errors.some((inner) => AiRetryError.isInstance(inner))
  ) {
    return error.errors.map((inner) => ({ model, error: inner }));
  }

  const cause = (error as { cause?: unknown } | null | undefined)?.cause;
  if (AiRetryError.isInstance(cause)) return cause.attempts;

  return undefined;
}

/**
 * The keys of the models an attempt ran on: the attempts of a retryable model
 * below it, or else its own model.
 */
function attemptModelKeys(attempt: AttemptLike): Array<string> {
  const inner = innerAttempts(attempt.error, attempt.model);
  return inner
    ? inner.flatMap((a) => attemptModelKeys(a))
    : [getModelKey(attempt.model)];
}

/**
 * Count how many of the given attempts ran against the given model. An attempt
 * through a retryable model counts for each model that retryable model tried.
 *
 * Accepts either layer's attempt type — only each attempt's model is read. A
 * union of the two rather than a structurally loosened `{ model }`, so `MODEL`
 * keeps its meaning. A bare `{ model: MODEL }` cannot work: a result attempt's
 * model is `LanguageModel` outright rather than generic, so it is never
 * assignable to an unresolved `MODEL`.
 */
export function countModelAttempts<MODEL extends AnyModel>(
  model: MODEL,
  attempts: ReadonlyArray<ModelRetryAttempt<MODEL> | CallRetryAttempt<MODEL>>,
): number {
  const modelKey = getModelKey(model);
  return attempts
    .flatMap((attempt) => attemptModelKeys(attempt))
    .filter((key) => key === modelKey).length;
}
