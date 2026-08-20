import { calculateExponentialBackoff } from './calculate-exponential-backoff.js';
import { countModelAttempts } from './count-model-attempts.js';
import type { CallRetryAttempt } from '../call/types.js';
import type { AnyModel, ModelRetryAttempt, Retry } from '../types.js';

/**
 * Resolve the exponential backoff delay to wait before a retry, or `undefined`
 * when the chosen retry sets no base delay. The exponent is the number of prior
 * attempts already made against the retry's model, so each repeat of the same
 * model waits longer: `baseDelay * backoffFactor ^ attempts`.
 */
export function resolveBackoffDelay<MODEL extends AnyModel>(
  /**
   * Only `model`, `delay` and `backoffFactor` are read, so the option shape is
   * left open — the same helper serves the model wrappers and the call-level
   * entry points, whose `options` are different types entirely.
   */
  retryModel: Retry<MODEL, unknown>,
  /** Either layer's attempts; only each attempt's model is read. */
  attempts: ReadonlyArray<ModelRetryAttempt<MODEL> | CallRetryAttempt<MODEL>>,
): number | undefined {
  if (!retryModel.delay) return undefined;

  const modelAttemptsCount = countModelAttempts(retryModel.model, attempts);
  return calculateExponentialBackoff(
    retryModel.delay,
    retryModel.backoffFactor,
    modelAttemptsCount,
  );
}
