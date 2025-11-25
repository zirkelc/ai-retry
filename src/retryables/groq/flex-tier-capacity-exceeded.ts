import { APICallError } from 'ai';
import { resolveRetryableOptions } from '../../internal/resolve-retryable-options.js';
import type {
  EmbeddingModelV2,
  LanguageModelV2,
  Retryable,
  RetryableOptions,
} from '../../types.js';
import { isErrorAttempt } from '../../utils.js';

/**
 * Custom: Flex Tier Capacity Exceeded
 * @see https://console.groq.com/docs/errors
 */
const ERROR_FLEX_TIER_CAPACITY_EXCEEDED = 498;

/**
 * Retryable for Groq's Flex Tier Capacity Exceeded error.
 * @see https://console.groq.com/docs/errors
 */
export function flexTierCapacityExceeded<
  MODEL extends LanguageModelV2 | EmbeddingModelV2,
>(model: MODEL, options?: RetryableOptions<MODEL>): Retryable<MODEL>;
export function flexTierCapacityExceeded<
  MODEL extends LanguageModelV2 | EmbeddingModelV2,
>(options: RetryableOptions<MODEL>): Retryable<MODEL>;
export function flexTierCapacityExceeded<
  MODEL extends LanguageModelV2 | EmbeddingModelV2,
>(
  modelOrOptions: MODEL | RetryableOptions<MODEL>,
  options?: RetryableOptions<MODEL> | undefined,
): Retryable<MODEL> {
  const resolvedOptions = resolveRetryableOptions(modelOrOptions, options);

  return (context) => {
    const { current } = context;

    if (isErrorAttempt(current)) {
      const { error } = current;

      if (APICallError.isInstance(error)) {
        const { statusCode } = error;

        if (statusCode === ERROR_FLEX_TIER_CAPACITY_EXCEEDED) {
          const model = resolvedOptions.model ?? current.model;
          return { model, ...resolvedOptions };
        }
      }
    }

    return undefined;
  };
}
