/**
 * Composable conditions for `retryableStreamText`.
 *
 *   import { error, httpStatus, finishReason, result, ... }
 *     from 'ai-retry/stream-text/conditions';
 *
 * The same names exist under `ai-retry/language-model/conditions`, for
 * `createRetryableModel`. They are not interchangeable, and the type system says
 * so: a condition built here is rejected by a model-level `retries` list and
 * vice versa, because the two layers see different results and different call
 * arguments.
 *
 * `result` is typed against what a stream can still be judged on — a generation
 * that ended without emitting anything. Past the first content part the attempt
 * belongs to the caller and no condition runs at all, so there is no text and no
 * tool calls to read here by construction. That is why a condition written for
 * `generateText` does not compile against this entry point: it reads fields a
 * pre-commit stream cannot have.
 */

import { createErrorAPI } from '../../../internal/conditions/error.js';
import type { ResolvableLanguageModel } from '../../../types.js';
import {
  createCallResultAPI,
  createFinishReasonAPI,
} from '../../conditions/result.js';
import type { StreamTextCommitResult } from '../types.js';

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

export const { result } = createCallResultAPI<
  ResolvableLanguageModel,
  StreamTextCommitResult
>();

export const { finishReason } = createFinishReasonAPI<
  ResolvableLanguageModel,
  StreamTextCommitResult
>();

export type { StreamTextCommitResult } from '../types.js';
export type { CallFinishReason, CallLanguageModelUsage } from '../../types.js';
