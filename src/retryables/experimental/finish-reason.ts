import type { ResolvableLanguageModel } from '../../types.js';
import type { Condition } from './condition.js';
import { result } from './result.js';

/**
 * Match the result's finish reason against one of the given values.
 *
 * @example
 * finishReason('content-filter')
 * finishReason('content-filter', 'length')
 */
export function finishReason<
  MODEL extends ResolvableLanguageModel = ResolvableLanguageModel,
>(...reasons: Array<string>): Condition<MODEL> {
  return result<MODEL>((res) =>
    reasons.includes(res.finishReason.unified as string),
  );
}
