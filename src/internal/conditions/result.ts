import { safeParseJSON } from '@ai-sdk/provider-utils';
import { fromJSONSchema } from 'zod';
import type {
  LanguageModelResult,
  ResolvableLanguageModel,
  RetryContext,
} from '../../types.js';
import { isResultAttempt } from '../guards.js';
import { Condition } from './condition.js';

/**
 * The unified finish reason produced by the AI SDK.
 */
export type FinishReason = LanguageModelResult['finishReason']['unified'];

/**
 * Build the result-side condition helpers (`result`, `finishReason`,
 * `schemaInvalid`) bound to a specific language-model family. Consumed
 * by `language-model/conditions/index.ts` so the entry point exposes
 * helpers whose `MODEL` generic is constrained to the right family.
 *
 * Result-based conditions are language-model only — embedding and image
 * results have a different shape and are not supported.
 */
export function createResultAPI<BOUND extends ResolvableLanguageModel>() {
  /**
   * Build a condition from a predicate over the current generate result.
   * The predicate runs only when the current attempt succeeded; error
   * attempts return false.
   *
   * **Important:** returns a `Condition`, not a `Retryable`. Call
   * `.switch()` or `.retry()` to plug it into `retries: [...]`.
   *
   * @example
   * result<MODEL>((res) => res.finishReason.unified === 'length')
   *   .switch({ model: fallback })
   */
  function result<MODEL extends BOUND = BOUND>(
    predicate: (
      res: LanguageModelResult,
      ctx: RetryContext<MODEL>,
    ) => boolean | Promise<boolean>,
  ): Condition<MODEL> {
    return new Condition<MODEL>(async (ctx) => {
      if (!isResultAttempt(ctx.current)) return false;
      return predicate(ctx.current.result, ctx);
    });
  }

  /**
   * Match the result's finish reason against one of the given values.
   *
   * **Important:** returns a `Condition`, not a `Retryable`. Call
   * `.switch()` or `.retry()` to plug it into `retries: [...]`.
   *
   * @example
   * result.finishReason('content-filter').switch({ model: fallback })
   * result.finishReason('length').retry({ maxAttempts: 3 })
   */
  result.finishReason = function finishReason<MODEL extends BOUND = BOUND>(
    ...reasons: Array<FinishReason>
  ): Condition<MODEL> {
    return result<MODEL>((res) => reasons.includes(res.finishReason.unified));
  };

  /**
   * Match the result's finish reason against one of the given values.
   *
   * **Important:** returns a `Condition`, not a `Retryable`. Call
   * `.switch()` or `.retry()` to plug it into `retries: [...]`.
   *
   * @example
   * finishReason('content-filter').switch({ model: fallback })
   * finishReason('length').retry({ maxAttempts: 3 })
   */
  function finishReason<MODEL extends BOUND = BOUND>(
    ...reasons: Array<FinishReason>
  ): Condition<MODEL> {
    return result.finishReason<MODEL>(...reasons);
  }

  /**
   * Match when the result text fails JSON schema validation. The schema
   * is read from the call's `responseFormat`, which `Output.object()`
   * sets automatically. No-op when no schema is configured.
   *
   * **Important:** returns a `Condition`, not a `Retryable`. Call
   * `.switch()` or `.retry()` to plug it into `retries: [...]`.
   *
   * @example
   * schemaInvalid().switch({ model: fallback })
   * schemaInvalid().retry({ maxAttempts: 3 })
   */
  function schemaInvalid<MODEL extends BOUND = BOUND>(): Condition<MODEL> {
    return result<MODEL>(async (res, ctx) => {
      if (!isResultAttempt(ctx.current)) return false;
      const callOptions = ctx.current.options;
      const text = res.content
        .filter((part) => part.type === 'text')
        .map((part) => part.text)
        .join('');
      if (!text) return false;
      const responseFormat = callOptions.responseFormat;
      if (responseFormat?.type !== 'json' || !responseFormat.schema) {
        return false;
      }
      const schema = fromJSONSchema(responseFormat.schema);
      const parseResult = await safeParseJSON({ text, schema });
      return !parseResult.success;
    });
  }

  return { result, finishReason, schemaInvalid };
}
