/**
 * Experimental composable conditions bound to `LanguageModel`. For use
 * with `generateText`, `generateObject`, `streamText`, `streamObject`.
 *
 *   import { error, httpStatus, finishReason, ... }
 *     from 'ai-retry/experimental/language-model/retryables';
 */

import type { ResolvableLanguageModel } from '../../../types.js';
import { createErrorAPI } from '../../internal/error.js';
import { createResultAPI } from '../../internal/result.js';

/**
 * Conditions are bound to `ResolvableLanguageModel` (instance or
 * gateway string literal) so `.switch({ model: 'openai/gpt-5' })` is
 * accepted alongside `.switch({ model: openai('gpt-4o') })`.
 */
export const { error, httpStatus, timeout, aborted } =
  createErrorAPI<ResolvableLanguageModel>();
export const { result, finishReason, schemaInvalid } =
  createResultAPI<ResolvableLanguageModel>();
