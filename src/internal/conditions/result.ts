import { safeParseJSON } from '@ai-sdk/provider-utils';
import { fromJSONSchema } from 'zod';
import type {
  LanguageModelResult,
  ResolvableLanguageModel,
  ModelRetryContext,
} from '../../types.js';
import { isResultAttempt } from '../guards.js';
import { Condition } from './condition.js';

/**
 * The unified finish reason produced by the AI SDK.
 */
export type ModelFinishReason = LanguageModelResult['finishReason']['unified'];

/**
 * Build the result-side condition helpers (`result`, `finishReason`,
 * `schemaInvalid`) bound to a specific language-model family. Consumed
 * by `language-model/conditions/index.ts` so the entry point exposes
 * helpers whose `MODEL` generic is constrained to the right family.
 *
 * Result-based conditions are language-model only here: the embedding and image
 * model wrappers have no result branch, so a condition for those families would
 * silently never fire. The call-level functions do support all three.
 */
export function createResultAPI<BOUND extends ResolvableLanguageModel>() {
  /**
   * Build a condition from a predicate over the current generate result.
   * The predicate runs only when the current attempt succeeded; error
   * attempts return false.
   *
   * The result is provider-shaped — what the model returned, one layer below
   * `generateText`, not the result the caller receives. The call-level retry
   * functions have their own `result()` under
   * `ai-retry/call/<family>-model/conditions`, which hands over the entry
   * point's own result instead; the two are different types and are not
   * interchangeable.
   *
   * **Important:** returns a `Condition`, not a `ModelRetryable`. Call
   * `.switch()` or `.retry()` to plug it into `retries: [...]`.
   *
   * @example
   * result<MODEL>((res) => res.finishReason.unified === 'length')
   *   .switch({ model: fallback })
   */
  function result<MODEL extends BOUND = BOUND>(
    predicate: (
      res: LanguageModelResult,
      ctx: ModelRetryContext<MODEL>,
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
   * Reads the attempt's normalized finish reason rather than digging into the
   * result, so it matches identically whether the retry ran below the model
   * (where the reason arrives nested) or around the call (where it arrives
   * flat).
   *
   * **Important:** returns a `Condition`, not a `ModelRetryable`. Call
   * `.switch()` or `.retry()` to plug it into `retries: [...]`.
   *
   * @example
   * result.finishReason('content-filter').switch({ model: fallback })
   * result.finishReason('length').retry({ maxAttempts: 3 })
   */
  result.finishReason = function finishReason<MODEL extends BOUND = BOUND>(
    ...reasons: Array<ModelFinishReason>
  ): Condition<MODEL> {
    return new Condition<MODEL>(
      (ctx) =>
        isResultAttempt(ctx.current) &&
        reasons.includes(ctx.current.finishReason),
    );
  };

  /**
   * Match the result's finish reason against one of the given values.
   *
   * **Important:** returns a `Condition`, not a `ModelRetryable`. Call
   * `.switch()` or `.retry()` to plug it into `retries: [...]`.
   *
   * @example
   * finishReason('content-filter').switch({ model: fallback })
   * finishReason('length').retry({ maxAttempts: 3 })
   */
  function finishReason<MODEL extends BOUND = BOUND>(
    ...reasons: Array<ModelFinishReason>
  ): Condition<MODEL> {
    return result.finishReason<MODEL>(...reasons);
  }

  /**
   * Match when the result text fails JSON schema validation. The schema
   * is read from the call's `responseFormat`, which `Output.object()`
   * sets automatically. No-op when no schema is configured.
   *
   * **Important:** returns a `Condition`, not a `ModelRetryable`. Call
   * `.switch()` or `.retry()` to plug it into `retries: [...]`.
   *
   * @deprecated Only meaningful for `createRetryableModel`. It reads
   * `responseFormat` off the provider call options, which do not exist around
   * a call, so it never matches for the call-level retry functions.
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
