import type {
  embed,
  embedMany,
  generateImage,
  generateText,
  streamText,
  ToolSet,
} from 'ai';
import type {
  AnyModel,
  AnyResolvableModel,
  EmbeddingModel,
  LanguageModel,
  ResolvableModel,
  ResolvedModel,
  Retry,
} from '../types.js';

/**
 * Every type the call-level retry layer is described in: what an attempt looks
 * like, what a retryable decides against, and what a result condition judges.
 * The runtime that goes with them lives in `guards.ts` (narrowing) and
 * `retryable-calls.ts` (tagging).
 *
 * The model layer's equivalents are in `src/types.ts` and stay frozen.
 */

/* ------------------------------------------------------------------ *
 * The retry context
 * ------------------------------------------------------------------ */

/**
 * Deliberately a different type from the model-level `ModelRetryContext`, not merely
 * a different set of exports. The two layers see genuinely different things — a
 * call-level attempt holds the entry point's own arguments and its own result,
 * a model-level one holds the provider's — and if both produced the same context
 * type, a condition written for either would silently typecheck against the
 * other's `retries` list.
 */

/** The arguments a language-model call can have been issued with. */
type LanguageModelCallArgs =
  | Parameters<typeof generateText>[0]
  | Parameters<typeof streamText>[0];

/** The arguments an embedding call can have been issued with. */
type EmbeddingModelCallArgs =
  | Parameters<typeof embed>[0]
  | Parameters<typeof embedMany>[0];

/** The arguments an image call can have been issued with. */
type ImageModelCallArgs = Parameters<typeof generateImage>[0];

/**
 * Maps a model family to the arguments its call-level entry points take.
 *
 * A union across the family's entry points, so whatever they share (`headers`
 * and `providerOptions` everywhere, `prompt` for language) reads directly, and
 * anything one of them owns needs narrowing.
 */
export type CallArgs<MODEL extends AnyModel> = MODEL extends LanguageModel
  ? LanguageModelCallArgs
  : MODEL extends EmbeddingModel
    ? EmbeddingModelCallArgs
    : ImageModelCallArgs;

/** A call-level attempt that failed with an error. */
export type CallRetryErrorAttempt<MODEL extends AnyModel> = {
  type: 'error';
  error: unknown;
  result?: undefined;
  /** The model this attempt was issued against. */
  model: MODEL;
  /** The arguments this attempt was issued with, overrides already applied. */
  options: CallArgs<MODEL>;
};

/**
 * A call-level attempt that produced a result which can still be failed over.
 *
 * For a stream that means it ended before emitting any content; past the first
 * content part the attempt belongs to the caller and never reaches a condition.
 */
export type CallRetryResultAttempt<MODEL extends AnyModel> = {
  type: 'result';
  error?: undefined;
  /** The entry point's own result, tagged with the operation that produced it. */
  result: CallResult<MODEL>;
  /** The model this attempt was issued against. */
  model: MODEL;
  /** The arguments this attempt was issued with, overrides already applied. */
  options: CallArgs<MODEL>;
};

/** A call-level attempt, with either an error or a judgeable result. */
export type CallRetryAttempt<MODEL extends AnyModel> =
  | CallRetryErrorAttempt<MODEL>
  | CallRetryResultAttempt<MODEL>;

/**
 * The context passed to a call-level retryable, with the attempt that triggered
 * the decision and every attempt made so far.
 */
export type CallRetryContext<MODEL extends AnyResolvableModel> = {
  /** The attempt that triggered this decision. */
  current: CallRetryAttempt<ResolvedModel<MODEL>>;
  /** Every attempt made so far, including the current one. */
  attempts: Array<CallRetryAttempt<ResolvedModel<MODEL>>>;
};

/**
 * A function that decides whether a call-level attempt should be retried, and
 * with which model.
 */
export type CallRetryable<MODEL extends AnyResolvableModel, INPUT = never> = (
  context: CallRetryContext<MODEL>,
) => Retry<MODEL, INPUT> | Promise<Retry<MODEL, INPUT> | undefined> | undefined;

/**
 * The configured call-level retry handlers.
 *
 * `INPUT` is the shape `Retry.options` is checked against — the entry point's
 * own arguments, not provider call options.
 */
export type CallRetries<MODEL extends AnyModel, INPUT> = Array<
  | CallRetryable<ResolvableModel<MODEL>, INPUT>
  | Retry<ResolvableModel<MODEL>, INPUT>
  | ResolvableModel<MODEL>
>;

/**
 * The context passed to `onFailure` when a call terminally fails: no retry
 * matched, every candidate was tried, or the caller cancelled.
 */
export type CallFailureContext<MODEL extends AnyResolvableModel> = {
  /** The final attempt that failed. */
  current: CallRetryErrorAttempt<ResolvedModel<MODEL>>;
  /** Every attempt made, including the final failed one. */
  attempts: Array<CallRetryAttempt<ResolvedModel<MODEL>>>;
  /**
   * The error surfaced to the caller. A `RetryError` wrapping every attempt
   * error when more than one attempt was made, otherwise the raw error.
   */
  error: unknown;
};

/* ------------------------------------------------------------------ *
 * The results
 * ------------------------------------------------------------------ */

/**
 * The results a call-level retry can judge, in the shape the AI SDK entry
 * points actually return.
 *
 * A retry running *below* a model sees the provider's result, so the model-level
 * conditions are written against that shape. A call-level retry never sees one:
 * it holds whatever the entry point returned. Rather than translate that back
 * into a provider result — lossy in one direction and a permanent drift surface
 * in the other — the result is passed through as-is, tagged with the operation
 * that produced it.
 *
 * One family can be reached through more than one entry point, so each family's
 * result is a **discriminated union** over `operation`. Whatever every member of
 * a family shares is readable directly — `finishReason` and `usage` for
 * language, `usage` and `response` for embedding — and anything specific to one
 * operation needs a guard:
 *
 * ```ts
 * result((res) => {
 *   if (res.finishReason === 'content-filter') return true;
 *   if (isGenerateTextResult(res)) return res.text.length < 10;
 *   return false;
 * });
 * ```
 *
 * The guards narrow within whatever result type the condition was given, so a
 * tool set has to be named at the condition — `result<typeof tools>(...)` — for
 * the tool calls to come out typed. Naming it on the guard has no effect.
 */

/** What `generateText` resolves to, with its generics left at their bounds. */
type SdkGenerateTextResult = Awaited<ReturnType<typeof generateText>>;

/**
 * The finish reason as the SDK entry points report it — flat, where a provider
 * reports it nested under `finishReason.unified`.
 *
 * Derived from the SDK's own result so the two members of the language union
 * are guaranteed to agree, which is what keeps `finishReason` readable without
 * narrowing.
 */
export type CallFinishReason = SdkGenerateTextResult['finishReason'];

/** Token usage as the SDK entry points report it. */
export type CallLanguageModelUsage = SdkGenerateTextResult['usage'];

/**
 * A completed `generateText`, exactly as the caller receives it.
 *
 * `TOOLS` is not inferred from the call — a condition is written against the
 * family, not against one call site, so there is nothing to infer it from. It is
 * whatever the condition names, and nothing checks that against the tools the
 * call was actually issued with; the contract is a cast's.
 */
export type GenerateTextResultInfo<TOOLS extends ToolSet = ToolSet> = {
  operation: 'generateText';
} & Awaited<ReturnType<typeof generateText<TOOLS>>>;

/**
 * A `streamText` that finished without ever emitting content.
 *
 * Deliberately not the SDK's own `StreamTextResult`: every field on that is a
 * promise that settles only once the stream has been consumed, and consuming it
 * is precisely what a pre-commit judgement must not do. What is here is read
 * off the stream's terminal parts instead.
 *
 * There is no content, and that is a fact rather than an omission — any content
 * part would have committed the attempt and put it beyond retry.
 */
export type StreamTextResultInfo = {
  operation: 'streamText';
  finishReason: CallFinishReason;
  usage: CallLanguageModelUsage;
  providerMetadata: SdkGenerateTextResult['providerMetadata'];
};

/** A completed `embed`, exactly as the caller receives it. */
export type EmbedResultInfo = {
  operation: 'embed';
} & Awaited<ReturnType<typeof embed>>;

/** A completed `embedMany`, exactly as the caller receives it. */
export type EmbedManyResultInfo = {
  operation: 'embedMany';
} & Awaited<ReturnType<typeof embedMany>>;

/** A completed `generateImage`, exactly as the caller receives it. */
export type GenerateImageResultInfo = {
  operation: 'generateImage';
} & Awaited<ReturnType<typeof generateImage>>;

/**
 * Everything a language-model call can produce that is still judgeable.
 * `finishReason`, `usage` and `providerMetadata` are common to both members.
 */
export type CallLanguageModelResult<TOOLS extends ToolSet = ToolSet> =
  | GenerateTextResultInfo<TOOLS>
  | StreamTextResultInfo;

/**
 * Everything an embedding call can produce. `usage`, `response` and
 * `providerMetadata` are common to both members; the embeddings themselves are
 * singular or plural depending on the entry point, so they need a guard.
 */
export type CallEmbeddingModelResult = EmbedResultInfo | EmbedManyResultInfo;

/**
 * Everything an image call can produce. One entry point, one member — no guard
 * is needed to read it.
 */
export type CallImageModelResult = GenerateImageResultInfo;

/**
 * Maps a model family to the results its call-level entry points produce.
 *
 * The result is a function of the model alone, which is what lets the call-level
 * retry context carry it without a generic of its own.
 */
export type CallResult<MODEL extends AnyModel> = MODEL extends LanguageModel
  ? CallLanguageModelResult
  : MODEL extends EmbeddingModel
    ? CallEmbeddingModelResult
    : CallImageModelResult;
