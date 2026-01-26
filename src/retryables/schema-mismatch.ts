import { safeParseJSON } from '@ai-sdk/provider-utils';
import { fromJSONSchema } from 'zod';
import type {
  ResolvableLanguageModel,
  Retryable,
  RetryableOptions,
} from '../types.js';
import { isResultAttempt } from '../utils.js';

/**
 * Fallback to a different model if the response does not match the expected JSON schema.
 *
 * Validates the response text against the JSON schema from `responseFormat.schema`
 * (set automatically when using `Output.object()`) and retries with a different model
 * if validation fails.
 *
 * @example
 * ```ts
 * const result = await generateText({
 *   model: createRetryable({
 *     model: primaryModel,
 *     retries: [schemaMismatch(fallbackModel)],
 *   }),
 *   output: Output.object({ schema: z.object({ name: z.string() }) }),
 *   prompt: `Generate a person`,
 * });
 * ```
 */
export function schemaMismatch<MODEL extends ResolvableLanguageModel>(
  model: MODEL,
  options?: RetryableOptions<MODEL>,
): Retryable<MODEL> {
  return async (context) => {
    /**
     * Only handle result attempts
     */
    if (!isResultAttempt(context.current)) return undefined;

    const { result, options: callOptions } = context.current;

    /** Extract text from content */
    const text = result.content
      .filter((part) => part.type === `text`)
      .map((part) => part.text)
      .join(``);
    if (!text) return undefined;

    /** Auto-detect schema from responseFormat (set by Output.object()) */
    const responseFormat = callOptions.responseFormat;
    if (responseFormat?.type !== `json` || !responseFormat.schema) {
      return undefined;
    }

    const schema = fromJSONSchema(responseFormat.schema);
    const parseResult = await safeParseJSON({ text, schema });
    if (parseResult.success) return undefined;

    return { model, maxAttempts: 1, ...options };
  };
}
