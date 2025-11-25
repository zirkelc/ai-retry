import type {
  LanguageModelV2,
  LanguageModelV2FinishReason,
} from '@ai-sdk/provider';
import type { Retryable, RetryableOptions } from '../types.js';
import { isResultAttempt } from '../utils.js';

interface FinishReasonOptions<MODEL extends LanguageModelV2>
  extends RetryableOptions<MODEL> {
  /**
   * Finish reason(s) to trigger retry.
   * Common retryable reasons: 'unknown', 'error', 'content-filter', 'other'
   */
  reasons: LanguageModelV2FinishReason | LanguageModelV2FinishReason[];
}

/**
 * Retry when specific finish reasons are encountered.
 *
 * Works with both generateText and streamText, but for streamText only retries
 * if no content was streamed (finish reason received before any content parts).
 *
 * This is particularly useful for handling cases where the model fails to generate
 * useful content, such as:
 * - 'unknown': Something went wrong, typically no content generated
 * - 'error': Explicit error, likely no or incomplete content
 * - 'content-filter': Content blocked, typically empty/minimal content
 * - 'other': Provider-specific reason, might indicate no content
 *
 * @example
 * ```ts
 * // Retry on unknown finish reasons
 * createRetryable({
 *   model: openai('gpt-4'),
 *   retries: [
 *     finishReason(anthropic('claude-3-5-sonnet'), {
 *       reasons: 'unknown',
 *       maxAttempts: 1
 *     })
 *   ]
 * })
 * ```
 *
 * @example
 * ```ts
 * // Retry on multiple problematic finish reasons
 * createRetryable({
 *   model: openai('gpt-4'),
 *   retries: [
 *     finishReason(fallback, {
 *       reasons: ['unknown', 'error', 'content-filter'],
 *       maxAttempts: 2,
 *       delay: 1000
 *     })
 *   ]
 * })
 * ```
 */
export function finishReason<MODEL extends LanguageModelV2>(
  model: MODEL,
  options: FinishReasonOptions<MODEL>,
): Retryable<MODEL> {
  return (context) => {
    const { current } = context;

    if (isResultAttempt(current)) {
      const { result } = current;
      const targetReasons = Array.isArray(options.reasons)
        ? options.reasons
        : [options.reasons];

      if (targetReasons.includes(result.finishReason)) {
        const { reasons, ...retryOptions } = options;
        return { model, ...retryOptions };
      }
    }

    return undefined;
  };
}
