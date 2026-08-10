import type {
  embed,
  embedMany,
  generateImage,
  generateText,
  streamText,
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
 * like and what a retryable decides against.
 *
 * What a *result* condition judges is deliberately not here. It belongs to the
 * entry point, not to the layer, so each one declares its own commit result in
 * its own folder and this file only ever carries it as `COMMIT`.
 *
 * The model layer's equivalents are in `src/types.ts` and stay frozen.
 */

/* ------------------------------------------------------------------ *
 * The commit result
 * ------------------------------------------------------------------ *
 *
 * `COMMIT`, throughout this file, is what a result condition of an entry point
 * judges: the outcome as it stands at the moment the attempt would commit,
 * while failing over is still possible.
 *
 * For most entry points that is simply the result the caller receives, which is
 * why it defaults to it. It is a parameter rather than a fixed type because
 * that does not always hold: `streamText` returns before anything has been
 * generated, and every field of its result is a promise that settles only once
 * the stream has been consumed — consuming it being precisely what a pre-commit
 * judgement must not do. What conditions see there is read off the stream's
 * terminal parts instead.
 *
 * Any entry point whose result cannot be inspected without spending it needs
 * the same treatment, so this is the general seam rather than one function's
 * exception. Each entry point declares its own in its own folder; nothing about
 * which is which lives here.
 */

/**
 * The unified finish reason as the SDK entry points report it — flat, where a
 * provider reports it nested under `finishReason.unified`.
 *
 * Shared rather than per-entry-point because the retry loop reads it off
 * whatever an attempt produced, to report on the attempt span.
 */
export type CallFinishReason = Awaited<
  ReturnType<typeof generateText>
>['finishReason'];

/** Token usage as the SDK language entry points report it. */
export type CallLanguageModelUsage = Awaited<
  ReturnType<typeof generateText>
>['usage'];

/* ------------------------------------------------------------------ *
 * The retry context
 * ------------------------------------------------------------------ */

/**
 * Deliberately a different type from the model-level `ModelRetryContext`, not
 * merely a different set of exports. The two layers see genuinely different
 * things — a call-level attempt holds the entry point's own arguments and its
 * own result, a model-level one holds the provider's — and if both produced the
 * same context type, a condition written for either would silently typecheck
 * against the other's `retries` list.
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
 * Still keyed on the family rather than on the entry point, unlike the commit
 * result: a union of argument objects needs no discriminant to be useful, since
 * whatever the entry points share (`headers` and `providerOptions` everywhere,
 * `prompt` for language) reads directly off it.
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
export type CallRetryResultAttempt<MODEL extends AnyModel, COMMIT = unknown> = {
  type: 'result';
  error?: undefined;
  /** The outcome as it stands while failing over is still possible. */
  result: COMMIT;
  /** The model this attempt was issued against. */
  model: MODEL;
  /** The arguments this attempt was issued with, overrides already applied. */
  options: CallArgs<MODEL>;
};

/** A call-level attempt, with either an error or a judgeable result. */
export type CallRetryAttempt<MODEL extends AnyModel, COMMIT = unknown> =
  | CallRetryErrorAttempt<MODEL>
  | CallRetryResultAttempt<MODEL, COMMIT>;

/**
 * The context passed to a call-level retryable, with the attempt that triggered
 * the decision and every attempt made so far.
 *
 * `COMMIT` defaults to `unknown`, which is what a condition that never reads the
 * result — every error condition — is built against. Because the context sits in
 * a condition's parameter position, that default makes such a condition
 * assignable to *every* entry point's `retries` list, while one built against a
 * specific commit result is assignable only where that result is actually
 * produced.
 */
export type CallRetryContext<
  MODEL extends AnyResolvableModel,
  COMMIT = unknown,
> = {
  /** The attempt that triggered this decision. */
  current: CallRetryAttempt<ResolvedModel<MODEL>, COMMIT>;
  /** Every attempt made so far, including the current one. */
  attempts: Array<CallRetryAttempt<ResolvedModel<MODEL>, COMMIT>>;
};

/**
 * A function that decides whether a call-level attempt should be retried, and
 * with which model.
 */
export type CallRetryable<
  MODEL extends AnyResolvableModel,
  INPUT = never,
  COMMIT = unknown,
> = (
  context: CallRetryContext<MODEL, COMMIT>,
) => Retry<MODEL, INPUT> | Promise<Retry<MODEL, INPUT> | undefined> | undefined;

/**
 * The configured call-level retry handlers.
 *
 * `INPUT` is the shape `Retry.options` is checked against — the entry point's
 * own arguments, not provider call options.
 */
export type CallRetries<
  MODEL extends AnyModel,
  INPUT,
  COMMIT = unknown,
> = Array<
  | CallRetryable<ResolvableModel<MODEL>, INPUT, COMMIT>
  | Retry<ResolvableModel<MODEL>, INPUT>
  | ResolvableModel<MODEL>
>;

/**
 * The context passed to `onFailure` when a call terminally fails: no retry
 * matched, every candidate was tried, or the caller cancelled.
 */
export type CallFailureContext<
  MODEL extends AnyResolvableModel,
  COMMIT = unknown,
> = {
  /** The final attempt that failed. */
  current: CallRetryErrorAttempt<ResolvedModel<MODEL>>;
  /** Every attempt made, including the final failed one. */
  attempts: Array<CallRetryAttempt<ResolvedModel<MODEL>, COMMIT>>;
  /**
   * The error surfaced to the caller. A `RetryError` wrapping every attempt
   * error when more than one attempt was made, otherwise the raw error.
   */
  error: unknown;
};
