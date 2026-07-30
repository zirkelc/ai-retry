import { delay } from '@ai-sdk/provider-utils';
import { BaseRetryableModel } from '../../internal/base-retryable-model.js';
import { evaluateError } from '../../internal/evaluate-error.js';
import { isErrorAttempt } from '../../internal/guards.js';
import { resolveBackoffDelay } from '../../internal/resolve-backoff-delay.js';
import { resolveModel } from '../../internal/resolve-model.js';
import { createRetryTelemetry } from '../../internal/telemetry.js';
import type {
  CallOptions,
  EmbeddingModel,
  FailureContext,
  ImageModel,
  LanguageModel,
  OnRetryOverrides,
  ProviderOptions,
  Reset,
  ResolvableModel,
  Retries,
  Retry,
  RetryableModelOptions,
  RetryAttempt,
  RetryCallOptions,
  RetryContext,
  RetryTelemetrySettings,
} from '../../types.js';

/**
 * Any model kind the driver can loop over. Defaults to `LanguageModel`, which
 * is the only kind with a call-level entry point today (`streamText`), so the
 * unparametrized aliases (`RetryCallAttempt`, `RetryableCallOptions`, …) stay
 * language-model-shaped for existing callers.
 */
type AnyModel = LanguageModel | EmbeddingModel | ImageModel;

/**
 * The per-attempt inputs handed to the call function. The retryable owns the
 * model selection and the per-attempt deadline so that each (re-)run gets a
 * fresh `abortSignal`; the call function only has to wire these into whatever
 * it invokes (e.g. an AI SDK `streamText`/`generateText` call).
 */
export type RetryCallAttempt<MODEL extends AnyModel = LanguageModel> = {
  /** Resolved model instance to use for this attempt. */
  model: MODEL;
  /** 1-based attempt number. */
  attempt: number;
  /**
   * The deadline for this attempt in milliseconds, if one applies: the matched
   * `Retry.timeout`, or `RetryCallRunOptions.timeout` for the first attempt.
   * Each attempt gets its own value, so a re-run starts from a fresh deadline.
   *
   * Apply it however the underlying call takes a deadline: pass it to a call
   * with its own timeout option (`generateText({ timeout })`), or build an
   * `AbortSignal.timeout(timeout)` from it for a call that only takes a signal.
   */
  timeout: number | undefined;
  /**
   * The caller's cancellation signal (`RetryCallRunOptions.abortSignal`), passed
   * through unchanged — forward it to propagate a genuine caller cancel. It
   * carries no deadline; the per-attempt timeout is {@link RetryCallAttempt.timeout}.
   */
  abortSignal: AbortSignal | undefined;
  /**
   * Per-attempt call option overrides to apply on top of the call's own
   * options (from `Retry.options` and any `onRetry` return value).
   */
  options: RetryCallOptions<MODEL>;
};

/**
 * Options that influence a single `run` invocation.
 */
export type RetryCallRunOptions = {
  /** Genuine caller cancellation signal, composed into every attempt. */
  abortSignal?: AbortSignal;
  /**
   * Deadline in milliseconds for the first attempt. Subsequent attempts use
   * their matched `Retry.timeout`. Creating the deadline here (rather than
   * letting the caller bake it into the underlying call) is what lets a re-run
   * start from a fresh signal.
   */
  timeout?: number;
};

/**
 * The attempt the driver committed to: the one whose call function returned,
 * after which no further fail-over is possible.
 *
 * Deliberately distinct from the model-level `SuccessAttempt`, which carries a
 * model result and full call options. Neither has an equivalent here: the
 * result is the call function's own and reaches the caller through `run`, and
 * the options are only the per-attempt overrides, since the driver has no call
 * options of its own.
 */
export type RetryCallCommitAttempt<MODEL extends AnyModel = LanguageModel> = {
  type: 'commit';
  /** The model whose attempt committed. */
  model: MODEL;
  /** The per-attempt overrides applied to the committed attempt. */
  options: RetryCallOptions<MODEL>;
};

/**
 * The context passed to `onCommit`, with the committed attempt and the
 * attempts that were retried before it.
 */
export type RetryCallCommitContext<MODEL extends AnyModel = LanguageModel> = {
  /** The attempt that committed. */
  current: RetryCallCommitAttempt<MODEL>;
  /**
   * The preceding attempts that were retried, in order. Empty when the first
   * attempt committed. The committed attempt is `current` and is not repeated
   * here.
   */
  attempts: Array<RetryAttempt<MODEL>>;
};

/**
 * The driver returned by {@link createRetryableCall}. Invoke it with a function
 * that performs one attempt; it loops over the configured retries until the
 * function returns (the result is passed through unchanged) or no retry
 * matches (the error is thrown, wrapped in a `RetryError` if more than one
 * attempt was made).
 */
export type RetryCall<MODEL extends AnyModel = LanguageModel> = <RESULT>(
  fn: (attempt: RetryCallAttempt<MODEL>) => Promise<RESULT>,
  runOptions?: RetryCallRunOptions,
) => Promise<RESULT>;

/**
 * Options for {@link createRetryableCall}.
 *
 * Mirrors the subset of `RetryableModelOptions` that applies to a generic
 * retry loop.
 */
export interface RetryableCallOptions<MODEL extends AnyModel = LanguageModel> {
  /** Base model used for the first attempt (resolved on first use). */
  model: ResolvableModel<MODEL>;
  /** Retry handlers / fallback models, evaluated on each error. */
  retries: Retries<MODEL>;
  disabled?: boolean | (() => boolean);
  /**
   * Controls when to reset back to the base model after a successful retry.
   *
   * @default 'after-request'
   */
  reset?: Reset;
  /**
   * Experimental. Can change in patch versions without warning.
   *
   * Telemetry configuration. When enabled, emits OpenTelemetry spans for retry
   * operations and attempts. Requires `@opentelemetry/api`.
   */
  experimental_telemetry?: RetryTelemetrySettings;
  onError?: (context: RetryContext<MODEL>) => void;
  /**
   * Called after a retry has been decided and the next model selected, but
   * before the retry call is issued. May return partial overrides for the
   * upcoming attempt.
   */
  onRetry?: (
    context: RetryContext<MODEL>,
  ) => void | OnRetryOverrides<MODEL> | Promise<void | OnRetryOverrides<MODEL>>;
  /**
   * Called once an attempt commits: the call function returned, so the driver
   * has locked that attempt in and will not fail over again. Reports the model
   * that handled it and the attempts that were retried before it.
   *
   * Not named `onSuccess`, because how much has to succeed before the call
   * function returns is the call function's choice, not the driver's. A
   * `generateText` call has fully completed by then; a `streamText` call has
   * only produced its result object, and whatever it goes on to stream — or
   * fail with — is past the driver's reach. The stream wrapper moves the
   * boundary to the first content part, which is as late as anything can still
   * fail over.
   *
   * The result is not reported here: the driver never inspects it, and the
   * caller receives it, correctly typed, from `run`.
   */
  onCommit?: (context: RetryCallCommitContext<MODEL>) => void;
  /**
   * Called once when the attempts are exhausted without committing: no retry
   * matched, every candidate was tried, the caller's signal was already
   * aborted, or the caller aborted during a backoff delay. `context.error` is
   * the error the run rejects with, and `context.current` the attempt that
   * failed last. The counterpart to `onCommit`.
   *
   * Reports attempt failures, so it stays silent for a rejection that no
   * attempt caused — a callback of your own throwing, or a retryable throwing
   * before the first attempt was recorded. Those still reject the run; there
   * is simply no failed attempt to hand over. Also silent when retries are
   * disabled.
   */
  onFailure?: (context: FailureContext<MODEL>) => void;
}

/**
 * {@link RetryableCallOptions} once the base model has been resolved from its
 * gateway-string form to an instance.
 */
type ResolvedCallOptions<MODEL extends AnyModel> = Omit<
  RetryableCallOptions<MODEL>,
  'model'
> & { model: MODEL };

/**
 * Resolve the per-attempt option overrides handed to the call function.
 *
 * Per-field precedence (highest → lowest):
 *   1. `onRetryOverrides.options.<field>`
 *   2. `currentRetry.options.<field>`
 *   3. `currentRetry.providerOptions` (deprecated top-level form, providerOptions only)
 */
function resolveRetryOptions<MODEL extends AnyModel>(
  currentRetry: Retry<MODEL> | undefined,
  onRetryOverrides: OnRetryOverrides<MODEL> | undefined,
): RetryCallOptions<MODEL> {
  const retryOptions = currentRetry?.options ?? {};
  const overrideOptions = onRetryOverrides?.options ?? {};
  const providerOptions =
    (overrideOptions as { providerOptions?: ProviderOptions })
      .providerOptions ??
    (retryOptions as { providerOptions?: ProviderOptions }).providerOptions ??
    currentRetry?.providerOptions;

  return {
    ...retryOptions,
    ...overrideOptions,
    ...(providerOptions ? { providerOptions } : {}),
  } as RetryCallOptions<MODEL>;
}

/**
 * Generic retry-loop driver. Unlike the model wrappers, this does not implement
 * a `LanguageModelV3`; it loops over an opaque async function and selects the
 * model + per-attempt deadline for each try. The call function decides what to
 * actually invoke, which keeps the driver independent of any specific AI SDK
 * entry point (`streamText`, `generateText`, …).
 *
 * Generic over the model kind; the resolved/conditional model types
 * (`ResolvedModel<MODEL>` from `findRetryModel`, `RetryAttempt<MODEL>`) collapse
 * to `MODEL` at runtime but TS can't prove it for a generic `MODEL`, so a few
 * casts bridge the gap — the same friction `evaluateError`/`findRetryModel`
 * already absorb internally.
 */
class RetryableCall<MODEL extends AnyModel> extends BaseRetryableModel<MODEL> {
  /**
   * The options under their call-level types. `BaseRetryableModel` stores them
   * under the model-level ones, which have no `onCommit` at all, so this is the
   * handle the loop reads the hooks through.
   */
  private readonly callOptions: ResolvedCallOptions<MODEL>;

  constructor(options: ResolvedCallOptions<MODEL>) {
    super(options as unknown as RetryableModelOptions<MODEL>);
    this.callOptions = options;
  }

  /**
   * Fire the `onCommit` callback for a call that returned.
   *
   * No cast, unlike {@link RetryableCall.emitFailure}: the commit context is
   * declared over `MODEL` directly, whereas the shared contexts are declared
   * over `ResolvedModel<MODEL>` — the same type at runtime, but not provably so
   * for a generic `MODEL`.
   */
  private emitCommit(
    current: RetryCallCommitAttempt<MODEL>,
    attempts: Array<RetryAttempt<MODEL>>,
  ) {
    this.callOptions.onCommit?.({ current, attempts });
  }

  /**
   * Fire the `onFailure` callback for a terminally failed call. The final
   * attempt (last entry of `attempts`) is surfaced as `current`.
   */
  private emitFailure(attempts: Array<RetryAttempt<MODEL>>, error: unknown) {
    if (!this.callOptions.onFailure) return;
    const current = attempts.at(-1);
    if (!current || !isErrorAttempt(current)) return;
    this.callOptions.onFailure({
      current,
      attempts,
      error,
    } as unknown as FailureContext<MODEL>);
  }

  async run<RESULT>(
    fn: (attempt: RetryCallAttempt<MODEL>) => Promise<RESULT>,
    runOptions?: RetryCallRunOptions,
  ): Promise<RESULT> {
    /**
     * Resolve the starting model (base or sticky).
     */
    const startModel = this.resolveStartModel();
    this.currentModel = startModel;

    /**
     * If retries are disabled, bypass retry machinery entirely. The first
     * attempt still receives a composed deadline from the run options.
     */
    if (this.isDisabled()) {
      return fn({
        model: startModel,
        attempt: 1,
        timeout: runOptions?.timeout,
        abortSignal: runOptions?.abortSignal,
        options: {} as RetryCallOptions<MODEL>,
      });
    }

    const recorder = await createRetryTelemetry(
      this.options.experimental_telemetry,
      {
        operation: 'call',
        genAiOperation: 'chat',
        provider: startModel.provider,
        modelId: startModel.modelId,
      },
    );

    /**
     * Track all attempts. The driver is purely error-based: a returned result
     * is terminal and never re-evaluated.
     */
    const attempts: Array<RetryAttempt<MODEL>> = [];

    /**
     * Track current retry configuration.
     */
    let currentRetry: Retry<MODEL> | undefined;

    let operationError: unknown;
    try {
      while (true) {
        /**
         * Call the onRetry handler if provided. Skip on the first attempt
         * since no previous attempt exists yet.
         */
        let onRetryOverrides: OnRetryOverrides<MODEL> | undefined;
        const previousAttempt = attempts.at(-1);
        if (previousAttempt) {
          const currentAttempt = {
            ...previousAttempt,
            model: this.currentModel,
          };

          const context = {
            current: currentAttempt,
            attempts: [...attempts],
          } as unknown as RetryContext<MODEL>;

          onRetryOverrides =
            (await this.callOptions.onRetry?.(context)) ?? undefined;
        }

        const attemptModel = this.currentModel;
        const attemptNumber = attempts.length + 1;

        /**
         * The deadline for this attempt: the matched retry's timeout, or the
         * first-attempt timeout from the run options. Surfaced as a number for
         * the call function to apply; the caller's signal is passed through
         * separately, so a re-run is not killed by an already-spent deadline.
         */
        const attemptTimeout = currentRetry?.timeout ?? runOptions?.timeout;
        const options = resolveRetryOptions(currentRetry, onRetryOverrides);

        recorder?.startAttempt({
          attempt: attemptNumber,
          provider: attemptModel.provider,
          modelId: attemptModel.modelId,
          timeoutMs: attemptTimeout,
        });

        /**
         * Only the call itself is guarded. Everything that runs once the
         * attempt has committed stays outside, so a throwing `onCommit`
         * handler cannot be mistaken for a failed attempt and re-run a call
         * that already succeeded.
         */
        let result: RESULT;
        try {
          result = await fn({
            model: attemptModel,
            attempt: attemptNumber,
            timeout: attemptTimeout,
            abortSignal: runOptions?.abortSignal,
            options,
          });
        } catch (error) {
          /**
           * Evaluate the failure. `options` is a minimal placeholder: the
           * driver has no prompt of its own (the call function owns it), and
           * error retryables only read `error`/`model`.
           */
          const evaluation = await evaluateError({
            error,
            model: attemptModel,
            options: {
              abortSignal: runOptions?.abortSignal,
              ...options,
            } as CallOptions<MODEL>,
            attempts,
            retries: this.options.retries,
            onError: this.callOptions.onError,
          });

          const retryModel = evaluation.retryModel as Retry<MODEL> | undefined;
          const finalError = evaluation.finalError;

          attempts.push(evaluation.attempt as RetryAttempt<MODEL>);

          /**
           * No retry matched. Surface the error, wrapped in a `RetryError`
           * when more than one attempt was made.
           */
          if (!retryModel) {
            recorder?.endAttempt({
              attempt: attemptNumber,
              outcome: 'failure',
              error,
            });
            throw finalError;
          }

          /**
           * If the caller's own signal is already aborted, any re-run would
           * forward that dead signal and abort instantly. Respect the cancel:
           * surface the error rather than fire a doomed retry.
           */
          if (runOptions?.abortSignal?.aborted) {
            recorder?.endAttempt({
              attempt: attemptNumber,
              outcome: 'failure',
              error,
            });
            throw error;
          }

          const calculatedDelay = resolveBackoffDelay(retryModel, attempts);

          recorder?.endAttempt({
            attempt: attemptNumber,
            outcome: 'retry',
            error,
            delayMs: calculatedDelay,
          });

          if (calculatedDelay !== undefined) {
            await delay(calculatedDelay, {
              abortSignal: runOptions?.abortSignal,
            });
          }

          this.currentModel = retryModel.model;
          currentRetry = retryModel;
          continue;
        }

        recorder?.endAttempt({ attempt: attemptNumber, outcome: 'success' });
        this.updateStickyModel(startModel);

        this.emitCommit(
          { type: 'commit', model: attemptModel, options },
          attempts,
        );

        return result;
      }
    } catch (error) {
      /**
       * Every way the loop can end without committing lands here: no retry
       * matched, the caller's signal was already aborted, the caller aborted
       * during a backoff delay, or a caller-supplied handler threw. Reporting
       * once at the boundary is what makes it impossible to reject without
       * telling `onFailure` and the operation span about it.
       */
      operationError = error;
      this.emitFailure(attempts, error);
      throw error;
    } finally {
      recorder?.endOperation({
        provider: this.currentModel.provider,
        modelId: this.currentModel.modelId,
        error: operationError,
      });
    }
  }
}

/**
 * Create a generic, entry-point-agnostic retry-loop driver.
 *
 * The returned function loops over the configured `retries`, selecting the
 * model and a fresh per-attempt deadline for each try, and invoking the
 * supplied call function. Because the call function performs the actual work,
 * the driver stays independent of `streamText`/`generateText` and can wrap any
 * call whose deadline must be re-established on each retry.
 *
 * Defaults to `LanguageModel` — the only model kind with a call-level entry
 * point today — but is generic over the model kind for future call-level
 * retries (embeddings, images).
 */
export function createRetryableCall<MODEL extends AnyModel = LanguageModel>(
  options: RetryableCallOptions<MODEL>,
): RetryCall<MODEL> {
  const model = resolveModel(options.model) as unknown as MODEL;
  const instance = new RetryableCall<MODEL>({
    ...options,
    model,
  });

  return (fn, runOptions) => instance.run(fn, runOptions);
}
