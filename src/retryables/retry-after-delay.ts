import { APICallError } from 'ai';
import { parseRetryHeaders } from '../parse-retry-headers.js';
import type {
  EmbeddingModelV2,
  LanguageModelV2,
  Retryable,
  RetryModel,
} from '../types.js';
import { isErrorAttempt } from '../utils.js';

const MAX_RETRY_AFTER_MS = 60_000;

type RetryAfterDelayOptions<MODEL extends LanguageModelV2 | EmbeddingModelV2> =
  Omit<RetryModel<MODEL>, 'model' | 'delay'> & {
    delay: number;
  };

/**
 * Retry with the same or a different model if the error is retryable with a delay.
 * Uses the `Retry-After` or `Retry-After-Ms` headers if present.
 * Otherwise uses the specified `delay`, with exponential backoff if `backoffFactor` is provided.
 */
export function retryAfterDelay<
  MODEL extends LanguageModelV2 | EmbeddingModelV2,
>(model: MODEL, options?: RetryAfterDelayOptions<MODEL>): Retryable<MODEL>;
export function retryAfterDelay<
  MODEL extends LanguageModelV2 | EmbeddingModelV2,
>(options: RetryAfterDelayOptions<MODEL>): Retryable<MODEL>;
export function retryAfterDelay<
  MODEL extends LanguageModelV2 | EmbeddingModelV2,
>(
  modelOrOptions?: MODEL | RetryAfterDelayOptions<MODEL>,
  options?: RetryAfterDelayOptions<MODEL>,
): Retryable<MODEL> {
  const model =
    modelOrOptions && 'delay' in modelOrOptions
      ? undefined
      : (modelOrOptions as MODEL | undefined);
  const opts =
    modelOrOptions && 'delay' in modelOrOptions ? modelOrOptions : options;

  if (!opts?.delay) {
    throw new Error('retryAfterDelay: delay is required');
  }

  return (context) => {
    const { current } = context;

    if (isErrorAttempt(current)) {
      const { error } = current;

      if (APICallError.isInstance(error) && error.isRetryable === true) {
        const targetModel = (model ?? current.model) as MODEL;

        const headerDelay = parseRetryHeaders(error.responseHeaders);
        if (headerDelay !== null) {
          return {
            model: targetModel,
            delay: Math.min(headerDelay, MAX_RETRY_AFTER_MS),
            maxAttempts: opts.maxAttempts,
            backoffFactor: opts.backoffFactor,
          };
        }

        return {
          model: targetModel,
          delay: opts.delay,
          maxAttempts: opts.maxAttempts,
          backoffFactor: opts.backoffFactor,
        };
      }
    }

    return undefined;
  };
}
