/**
 * Composable conditions bound to `LanguageModel`, for the call-level retry
 * functions `retryableGenerateText` and `retryableStreamText`.
 *
 *   import { error, httpStatus, finishReason, result, ... }
 *     from 'ai-retry/call/language-model/conditions';
 *
 * The same names exist under `ai-retry/language-model/conditions`, for
 * `createRetryableModel`. They are not interchangeable, and the type system says
 * so: a condition built here is rejected by a model-level `retries` list and
 * vice versa, because the two layers see different results and different call
 * arguments.
 */

import { createErrorAPI } from '../../../internal/conditions/error.js';
import type { ResolvableLanguageModel } from '../../../types.js';
import { createCallLanguageModelResultAPI } from '../../conditions/result.js';

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
export const { result, finishReason } =
  createCallLanguageModelResultAPI<ResolvableLanguageModel>();

export { isGenerateTextResult, isStreamTextResult } from '../../guards.js';
export type {
  CallFinishReason,
  CallLanguageModelResult,
  GenerateTextResultInfo,
  StreamTextResultInfo,
} from '../../types.js';
