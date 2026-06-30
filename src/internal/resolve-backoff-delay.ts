import { calculateExponentialBackoff } from './calculate-exponential-backoff.js';
import { countModelAttempts } from './count-model-attempts.js';
import type {
  EmbeddingModel,
  ImageModel,
  LanguageModel,
  Retry,
  RetryAttempt,
} from '../types.js';

/**
 * Resolve the exponential backoff delay to wait before a retry, or `undefined`
 * when the chosen retry sets no base delay. The exponent is the number of prior
 * attempts already made against the retry's model, so each repeat of the same
 * model waits longer: `baseDelay * backoffFactor ^ attempts`.
 */
export function resolveBackoffDelay<
  MODEL extends LanguageModel | EmbeddingModel | ImageModel,
>(
  retryModel: Retry<MODEL>,
  attempts: ReadonlyArray<RetryAttempt<MODEL>>,
): number | undefined {
  if (!retryModel.delay) return undefined;

  const modelAttemptsCount = countModelAttempts(retryModel.model, attempts);
  return calculateExponentialBackoff(
    retryModel.delay,
    retryModel.backoffFactor,
    modelAttemptsCount,
  );
}
