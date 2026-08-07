import type { ToolSet } from 'ai';
import type {
  CallEmbeddingModelResult,
  CallImageModelResult,
  CallLanguageModelResult,
  EmbedManyResultInfo,
  EmbedResultInfo,
  GenerateImageResultInfo,
  GenerateTextResultInfo,
  StreamTextResultInfo,
} from './types.js';

/**
 * Narrowing for the call-level result unions.
 *
 * Each family is reachable through one or more entry points, so its result is a
 * union discriminated by `operation`. These narrow it to one member. The
 * model layer's attempt guards are in `src/internal/guards.ts`.
 */

/**
 * Narrow a language-model result to a completed `generateText`.
 *
 * `TOOLS` is taken from the result being narrowed, not asserted here — pin it
 * at the condition (`result<typeof tools>(...)`) and the tool calls arrive
 * typed. Asserting it on the guard instead does nothing: the declared type of
 * the value being narrowed wins, so the assertion is silently ignored.
 *
 * @example
 * result<typeof tools>((res) =>
 *   isGenerateTextResult(res) && res.toolCalls.length === 0)
 */
export const isGenerateTextResult = <TOOLS extends ToolSet = ToolSet>(
  result: CallLanguageModelResult<TOOLS>,
): result is GenerateTextResultInfo<TOOLS> =>
  result.operation === 'generateText';

/**
 * Narrow a language-model result to a `streamText` that finished without
 * emitting content.
 *
 * @example
 * result((res) => isStreamTextResult(res) && res.usage.outputTokens === 0)
 */
export const isStreamTextResult = <TOOLS extends ToolSet = ToolSet>(
  result: CallLanguageModelResult<TOOLS>,
): result is StreamTextResultInfo => result.operation === 'streamText';

/**
 * Narrow an embedding result to a completed `embed` — a single value, so the
 * embedding is `embedding` rather than `embeddings`.
 *
 * @example
 * result((res) => isEmbedResult(res) && res.embedding.length === 0)
 */
export const isEmbedResult = (
  result: CallEmbeddingModelResult,
): result is EmbedResultInfo => result.operation === 'embed';

/**
 * Narrow an embedding result to a completed `embedMany`.
 *
 * @example
 * result((res) => isEmbedManyResult(res) && res.embeddings.length < expected)
 */
export const isEmbedManyResult = (
  result: CallEmbeddingModelResult,
): result is EmbedManyResultInfo => result.operation === 'embedMany';

/**
 * Narrow an image result to a completed `generateImage`.
 *
 * The image family has a single entry point, so this is never needed to read a
 * field — it exists for symmetry, and for code that is generic over families.
 */
export const isGenerateImageResult = (
  result: CallImageModelResult,
): result is GenerateImageResultInfo => result.operation === 'generateImage';
