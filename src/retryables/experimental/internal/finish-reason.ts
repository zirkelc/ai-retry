import type { ResolvableLanguageModel } from '../../../types.js';
import type { Condition } from './condition.js';
import { type FinishReason, result } from './result.js';

/**
 * Match the result's finish reason against one of the given values.
 * Thin wrapper around `result.finishReason(...)`.
 *
 * @example
 * finishReason('content-filter')
 * finishReason('content-filter', 'length')
 */
export function finishReason<
  MODEL extends ResolvableLanguageModel = ResolvableLanguageModel,
>(...reasons: Array<FinishReason>): Condition<MODEL> {
  return result.finishReason<MODEL>(...reasons);
}
