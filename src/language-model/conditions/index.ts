/**
 * Composable conditions bound to `LanguageModel`. For use with
 * `generateText`, `generateObject`, `streamText`, `streamObject`.
 *
 *   import { error, httpStatus, finishReason, ... }
 *     from 'ai-retry/language-model/conditions';
 */

import type { ResolvableLanguageModel } from '../../types.js';
import { createErrorAPI } from '../../internal/conditions/error.js';
import { createResultAPI } from '../../internal/conditions/result.js';

export { and } from '../../internal/conditions/and.js';
export { not } from '../../internal/conditions/not.js';
export { or } from '../../internal/conditions/or.js';

/**
 * Conditions are bound to `ResolvableLanguageModel` (instance or
 * gateway string literal) so `.switch({ model: 'openai/gpt-5' })` is
 * accepted alongside `.switch({ model: openai('gpt-4o') })`.
 */
export const { error, httpStatus, timeout, aborted } =
  createErrorAPI<ResolvableLanguageModel>();
export const { result, finishReason, schemaInvalid } =
  createResultAPI<ResolvableLanguageModel>();
