import type {
  embed,
  embedMany,
  generateImage,
  generateText,
  streamText,
} from 'ai';

/**
 * The arguments each entry point accepts, instantiated at their constraints.
 * Only used to derive the override types below — the public signatures keep
 * their own generics, since these instantiations are monomorphic.
 */
type GenerateTextArgs = Parameters<typeof generateText>[0];
type StreamTextArgs = Parameters<typeof streamText>[0];
type EmbedArgs = Parameters<typeof embed>[0];
type EmbedManyArgs = Parameters<typeof embedMany>[0];
type GenerateImageArgs = Parameters<typeof generateImage>[0];

/**
 * The arguments a retry may replace for the next attempt.
 *
 * Deliberately excludes `model` (the retry names it directly), `abortSignal`
 * and `timeout` (the retry loop owns the deadline, via `Retry.timeout`), and
 * anything structural such as `tools` — swapping the tool set mid-run would
 * invalidate the result type the caller was handed.
 *
 * A closed list rather than one derived from the entry point's arguments, and
 * that is the point: deriving would make every argument the SDK adds
 * overridable by default, including structural ones. Today that would already
 * admit `output`, `toolChoice`, `stopWhen`, `toolOrder` and `toolsContext`,
 * each of which changes the result or the tool loop the caller was promised —
 * plus a `telemetry` that collides with this library's own.
 *
 * The list is still checked against the real arguments: it is only ever used
 * through `Pick`, which constrains it to `keyof` — and to `keyof` *both*
 * language entry points, since the same list serves `generateText` and
 * `streamText`. A key that is misspelled, renamed by the SDK, or present on
 * only one of the two fails to compile.
 */
type LanguageOverrides =
  | 'prompt'
  | 'messages'
  | 'instructions'
  | 'system'
  | 'temperature'
  | 'maxOutputTokens'
  | 'topP'
  | 'topK'
  | 'presencePenalty'
  | 'frequencyPenalty'
  | 'seed'
  | 'stopSequences'
  | 'headers'
  | 'providerOptions'
  | 'maxRetries';

/**
 * Call arguments a retry may override for `retryableGenerateText`.
 *
 * These are the entry point's own arguments, not provider call options, so a
 * retry can rewrite the prompt in the shape the caller wrote it.
 */
export type GenerateTextInput = Partial<
  Pick<GenerateTextArgs, LanguageOverrides>
>;

/** Call arguments a retry may override for `retryableStreamText`. */
export type StreamTextInput = Partial<Pick<StreamTextArgs, LanguageOverrides>>;

/** Call arguments a retry may override for `retryableEmbed`. */
export type EmbedInput = Partial<
  Pick<EmbedArgs, 'value' | 'headers' | 'providerOptions' | 'maxRetries'>
>;

/** Call arguments a retry may override for `retryableEmbedMany`. */
export type EmbedManyInput = Partial<
  Pick<
    EmbedManyArgs,
    'values' | 'headers' | 'providerOptions' | 'maxRetries' | 'maxParallelCalls'
  >
>;

/** Call arguments a retry may override for `retryableGenerateImage`. */
export type GenerateImageInput = Partial<
  Pick<
    GenerateImageArgs,
    | 'prompt'
    | 'n'
    | 'size'
    | 'aspectRatio'
    | 'seed'
    | 'headers'
    | 'providerOptions'
    | 'maxRetries'
  >
>;
