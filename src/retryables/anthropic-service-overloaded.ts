import type { LanguageModelV2 } from '@ai-sdk/provider';
import { APICallError } from 'ai';
import type { Retryable, RetryModel } from '../create-retryable-model.js';
import { isErrorAttempt } from '../create-retryable-model.js';
import { isObject, isString } from '../utils.js';

/**
 * Type for Anthropic error responses.
 *
 * @see https://docs.claude.com/en/api/errors#error-shapes
 */
export type AnthropicErrorResponse = {
  type: 'error';
  error: {
    type: string;
    message: string;
  };
};

/**
 * Fallback if Anthropic returns an "overloaded" error with HTTP 200.
 *
 * ```
 * HTTP 200 OK
 * {"type":"error","error":{"type":"overloaded_error","message":"Overloaded"}}
 * ```
 *
 * @deprecated Use `serviceOverloaded` instead
 */
export function anthropicServiceOverloaded(
  model: LanguageModelV2,
  options?: Omit<RetryModel, 'model'>,
): Retryable {
  return (context) => {
    const { current } = context;

    if (isErrorAttempt(current)) {
      const { error } = current;

      // Anthropic returned a 529 status code
      if (APICallError.isInstance(error) && error.statusCode === 529) {
        return { model, maxAttempts: 1, ...options };
      }

      // Anthropic returned a 200 status code with an overloaded error in the body
      if (
        isObject(error) &&
        isString(error.type) &&
        error.type === 'overloaded_error'
      ) {
        return { model, maxAttempts: 1, ...options };
      }
    }

    return undefined;
  };
}
