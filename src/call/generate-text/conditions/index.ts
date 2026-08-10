/**
 * Composable conditions for `retryableGenerateText`.
 *
 *   import { error, httpStatus, finishReason, result, ... }
 *     from 'ai-retry/generate-text/conditions';
 *
 * The same names exist under `ai-retry/language-model/conditions`, for
 * `createRetryableModel`. They are not interchangeable, and the type system says
 * so: a condition built here is rejected by a model-level `retries` list and
 * vice versa, because the two layers see different results and different call
 * arguments.
 *
 * `result` is typed against this entry point's result alone — no union, no
 * discriminant, nothing to narrow. A condition that reads it is accepted only
 * where that result is actually produced; one that reads only the error, as
 * every helper below except `result` and `finishReason` does, fits any entry
 * point of this family.
 */

import type { ToolSet } from 'ai';
import type { Condition } from '../../../internal/conditions/condition.js';
import type { ResolvableLanguageModel } from '../../../types.js';
import { createErrorAPI } from '../../../internal/conditions/error.js';
import { resultCondition } from '../../conditions/result.js';
import type { CallFinishReason, CallRetryContext } from '../../types.js';
import type { GenerateTextCommitResult } from '../types.js';

export { and } from '../../../internal/conditions/and.js';
export { not } from '../../../internal/conditions/not.js';
export { or } from '../../../internal/conditions/or.js';

/**
 * Conditions are bound to `ResolvableLanguageModel` (instance or gateway
 * string literal) so `.switch({ model: 'openai/gpt-5' })` is accepted alongside
 * `.switch({ model: openai('gpt-4o') })`.
 */
export const { error, httpStatus, timeout, aborted } = createErrorAPI<
  ResolvableLanguageModel,
  'call'
>();

/**
 * Match the result's finish reason against one of the given values.
 *
 * Spelled out rather than destructured off the shared factory for the same
 * reason `result` is: the commit result is an SDK type whose own defaults
 * reference names `ai` does not export, so an inferred signature cannot be
 * written to a declaration file. Naming the return type keeps the alias intact.
 *
 * **Important:** returns a `Condition`, not a retryable. Call `.switch()` or
 * `.retry()` to plug it into `retry: [...]`.
 *
 * @example
 * finishReason('content-filter').switch({ model: fallback })
 * finishReason('length').retry({ maxAttempts: 3 })
 */
export function finishReason<
  MODEL extends ResolvableLanguageModel = ResolvableLanguageModel,
>(
  ...reasons: Array<CallFinishReason>
): Condition<MODEL, 'call', GenerateTextCommitResult> {
  return resultCondition<MODEL, GenerateTextCommitResult>((res) =>
    reasons.includes(res.finishReason),
  );
}

/**
 * Build a condition from a predicate over the completed generation. The
 * predicate runs only when the current attempt produced one; error attempts
 * return false.
 *
 * Declared here rather than taken from the shared factory because the result
 * carries a generic of its own: `TOOLS` names the tool set the tool calls
 * should be typed against, and has to be given at the condition since there is
 * no call site to infer it from. It is unchecked — nothing verifies it matches
 * the tools the call was issued with, the same contract as a cast.
 *
 * **Important:** returns a `Condition`, not a retryable. Call `.switch()` or
 * `.retry()` to plug it into `retry: [...]`.
 *
 * @example
 * result((res) => res.text.length < 10).switch({ model: fallback })
 *
 * @example
 * result<typeof tools>((res) => res.toolCalls.length === 0)
 *   .retry({ maxAttempts: 3 })
 */
export function result<
  TOOLS extends ToolSet = ToolSet,
  MODEL extends ResolvableLanguageModel = ResolvableLanguageModel,
>(
  predicate: (
    res: GenerateTextCommitResult<TOOLS>,
    ctx: CallRetryContext<MODEL, GenerateTextCommitResult<TOOLS>>,
  ) => boolean | Promise<boolean>,
): Condition<MODEL, 'call', GenerateTextCommitResult<TOOLS>> {
  return resultCondition<MODEL, GenerateTextCommitResult<TOOLS>>(predicate);
}

export type { GenerateTextCommitResult } from '../types.js';
export type { CallFinishReason, CallLanguageModelUsage } from '../../types.js';
