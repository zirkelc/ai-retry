import { safeParseJSON } from '@ai-sdk/provider-utils';
import { fromJSONSchema } from 'zod';
import type { ResolvableLanguageModel } from '../../types.js';
import { isResultAttempt } from '../../internal/guards.js';
import type { Condition } from './condition.js';
import { result } from './result.js';

/**
 * Match when the result text fails JSON schema validation. The schema is
 * read from the call's `responseFormat`, which `Output.object()` sets
 * automatically. No-op when no schema is configured.
 *
 * @example
 * schemaInvalid().switch({ model: fallback })
 */
export function schemaInvalid<
  MODEL extends ResolvableLanguageModel = ResolvableLanguageModel,
>(): Condition<MODEL> {
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
