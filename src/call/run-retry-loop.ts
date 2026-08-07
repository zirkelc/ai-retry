import { delay } from '@ai-sdk/provider-utils';
import { evaluateError } from '../internal/evaluate-error.js';
import { findRetryModel } from '../internal/find-retry-model.js';
import { isErrorAttempt } from '../internal/guards.js';
import { resolveBackoffDelay } from '../internal/resolve-backoff-delay.js';
import {
  type GatewayResolver,
  resolveModel,
} from '../internal/resolve-model.js';
import {
  createRetryTelemetry,
  type GenAiOperation,
} from '../internal/telemetry.js';
import type {
  AnyModel,
  AnyResolvableModel,
  OnRetryOverrides,
  ProviderOptions,
  Retry,
} from '../types.js';
import type {
  CallArgs,
  CallFailureContext,
  CallRetryAttempt,
  CallRetryContext,
  CallRetryResultAttempt,
} from './types.js';
import type { CallFinishReason, CallResult } from './types.js';
import type { CallRetryOptions } from './retry-arg.js';

/**
 * The subset of an entry point's arguments the loop itself reads. Everything
 * else is opaque and passed through.
 *
 * Not to be confused with `CallArgs`, which is the *whole* argument object of a
 * family's entry points, as a condition sees it on an attempt.
 */
export type RetryLoopArgs = {
  model: AnyResolvableModel;
  abortSignal?: AbortSignal;
  maxRetries?: number;
};

/**
 * The outcome of an attempt that returned rather than threw.
 *
 * - `committed` — the caller owns it now; nothing after this can be retried.
 * - `result` — there is a complete outcome to judge against result conditions,
 *   and failing over is still possible because the caller has seen nothing.
 */
export type Settled<INFO> =
  | { type: 'committed' }
  | { type: 'result'; result: INFO };

/**
 * How an entry point applies a per-attempt deadline to its arguments.
 *
 * `callerSignal` is the caller's own signal, unmodified — a strategy that
 * composes a deadline into `abortSignal` needs it, and one that has a
 * dedicated `timeout` argument ignores it.
 */
export type DeadlineStrategy<ARGS> = (
  args: ARGS,
  timeoutMs: number | undefined,
  callerSignal: AbortSignal | undefined,
) => ARGS;

/**
 * Everything that differs between the five entry points. The loop names none
 * of it directly, which is what keeps the retry logic single-sourced.
 */
export type EntryPoint<
  MODEL extends AnyModel,
  ARGS extends RetryLoopArgs,
  RESULT,
> = {
  /** Span name and `ai_retry.operation` attribute. */
  operation: string;
  /** Standard `gen_ai.operation.name` value for the underlying model call. */
  genAiOperation: GenAiOperation;
  /**
   * Resolves gateway model-id strings for this entry point's model family. A
   * bare string is ambiguous across families.
   */
  resolveGatewayModel: GatewayResolver;
  /** Issues one attempt. */
  call: (args: ARGS) => Promise<RESULT>;
  /** Applies the per-attempt deadline. */
  deadline: DeadlineStrategy<ARGS>;
  /**
   * Decides whether a returned result is terminal or still judgeable against
   * result conditions, and reports it in the shape conditions see — the entry
   * point's own result, tagged with the operation that produced it. Omitted
   * where a returned result is always terminal.
   *
   * Throwing here is indistinguishable from the call throwing, which is what
   * lets a stream that fails before its first content part reuse the entire
   * error path with no branch in the loop.
   */
  settle?: (
    result: RESULT,
    callerSignal: AbortSignal | undefined,
  ) => Promise<Settled<CallResult<MODEL>>>;
};

/** Whether the `disabled` switch is on for this call. */
const isDisabled = (disabled: boolean | (() => boolean) | undefined): boolean =>
  typeof disabled === 'function' ? disabled() : disabled === true;

/**
 * The finish reason a result carries, where the operation has one. Embeddings
 * and images do not, and report nothing rather than a placeholder.
 */
const finishReasonOf = (result: unknown): CallFinishReason | undefined =>
  (result as { finishReason?: CallFinishReason }).finishReason;

/**
 * Resolve the argument overrides for the upcoming attempt.
 *
 * Per-field precedence (highest → lowest):
 *   1. the `onRetry` return value
 *   2. `Retry.options`
 *   3. `Retry.providerOptions` (deprecated top-level form, providerOptions only)
 *
 * Anything not named here falls through to the call's own arguments, which the
 * loop spreads underneath.
 */
function resolveOverrides<MODEL extends AnyModel, INPUT>(
  currentRetry: Retry<MODEL, INPUT> | undefined,
  onRetryOverrides: OnRetryOverrides<MODEL, unknown> | undefined,
): Partial<INPUT> {
  const retryOptions = (currentRetry?.options ?? {}) as Partial<INPUT>;
  const overrideOptions = (onRetryOverrides?.options ?? {}) as Partial<INPUT>;
  const providerOptions =
    (overrideOptions as { providerOptions?: ProviderOptions })
      .providerOptions ??
    (retryOptions as { providerOptions?: ProviderOptions }).providerOptions ??
    currentRetry?.providerOptions;

  return {
    ...retryOptions,
    ...overrideOptions,
    ...(providerOptions ? { providerOptions } : {}),
  };
}

/**
 * Report a terminally failed call. The final attempt (last entry of `attempts`)
 * is surfaced as `current`; a rejection that no attempt caused has none, and
 * stays silent.
 */
function emitFailure<MODEL extends AnyModel, INPUT, OVERRIDE, RESULT>(
  options: CallRetryOptions<MODEL, INPUT, OVERRIDE, RESULT>,
  attempts: Array<CallRetryAttempt<MODEL>>,
  error: unknown,
): void {
  if (!options.onFailure) return;
  const current = attempts.at(-1);
  if (!current || !isErrorAttempt(current as any)) return;
  options.onFailure({
    current,
    attempts,
    error,
  } as unknown as CallFailureContext<MODEL>);
}

/**
 * The retry loop shared by every call-level entry point.
 *
 * Runs one attempt at a time: selects the model, applies the per-attempt
 * deadline, issues the call, and decides from the outcome whether to fail over.
 * Everything entry-point-specific lives in {@link EntryPoint}; everything
 * retry-specific lives here, once.
 *
 * Two properties are worth stating outright, because getting either wrong is
 * silent:
 *
 * - **The caller's signal is never conflated with ours.** It is read for the
 *   "already cancelled, do not fail over" check and handed to the deadline
 *   strategy separately from the composed per-attempt signal. If the two were
 *   merged, our own deadline would look like a caller cancel and kill fail-over.
 * - **The SDK's in-call retries are disabled by default.** Left at their
 *   default the entry point would re-issue the failing model several times
 *   before the loop ever saw the error, multiplying every deadline. A caller
 *   who sets `maxRetries` explicitly keeps it.
 */
export async function runRetryLoop<
  MODEL extends AnyModel,
  ARGS extends RetryLoopArgs,
  RESULT,
  INPUT,
  OVERRIDE,
>(input: {
  entryPoint: EntryPoint<MODEL, ARGS, RESULT>;
  args: ARGS;
  options: CallRetryOptions<MODEL, INPUT, OVERRIDE, RESULT>;
}): Promise<RESULT> {
  const { entryPoint, args, options } = input;

  /**
   * The caller's own cancellation signal, kept raw for the whole run.
   */
  const callerSignal = args.abortSignal;

  const baseModel = resolveModel(
    args.model,
    entryPoint.resolveGatewayModel,
  ) as MODEL;

  /**
   * Disabled: issue the call exactly as the caller wrote it, so the behavior
   * is indistinguishable from calling the entry point directly.
   */
  if (isDisabled(options.disabled)) {
    return entryPoint.call({ ...args, model: baseModel });
  }

  const recorder = await createRetryTelemetry(options.telemetry, {
    operation: entryPoint.operation,
    genAiOperation: entryPoint.genAiOperation,
    provider: baseModel.provider,
    modelId: baseModel.modelId,
  });

  const attempts: Array<CallRetryAttempt<MODEL>> = [];
  let currentModel = baseModel;
  let currentRetry: Retry<MODEL, INPUT> | undefined;

  let operationError: unknown;
  try {
    while (true) {
      /**
       * Ask for overrides for the upcoming attempt. Skipped on the first, where
       * there is no previous attempt to report.
       */
      let onRetryOverrides: OnRetryOverrides<MODEL, unknown> | undefined;
      const previousAttempt = attempts.at(-1);
      if (previousAttempt) {
        const context = {
          current: { ...previousAttempt, model: currentModel },
          attempts: [...attempts],
        } as unknown as CallRetryContext<MODEL>;

        onRetryOverrides = (await options.onRetry?.(context)) ?? undefined;
      }

      const attemptModel = currentModel;
      const attemptNumber = attempts.length + 1;
      const attemptTimeout = currentRetry?.timeout;

      const attemptArgs = entryPoint.deadline(
        {
          ...args,
          ...resolveOverrides(currentRetry, onRetryOverrides),
          model: attemptModel,
          maxRetries: args.maxRetries ?? 0,
        } as ARGS,
        attemptTimeout,
        callerSignal,
      );

      recorder?.startAttempt({
        attempt: attemptNumber,
        provider: attemptModel.provider,
        modelId: attemptModel.modelId,
        timeoutMs: attemptTimeout,
      });

      /**
       * Only the attempt itself is guarded. Everything that runs once the
       * outcome is known stays outside, so a throwing `onSuccess` cannot be
       * mistaken for a failed attempt and re-run a call that already succeeded.
       */
      let result: RESULT;
      let settled: Settled<CallResult<MODEL>>;
      try {
        result = await entryPoint.call(attemptArgs);
        settled = (await entryPoint.settle?.(result, callerSignal)) ?? {
          type: 'committed',
        };
      } catch (error) {
        const evaluation = await evaluateError({
          error,
          model: attemptModel,
          options: attemptArgs as unknown as CallArgs<MODEL>,
          attempts,
          retries: options.retries,
          onError: options.onError as unknown as (context: never) => void,
          resolve: entryPoint.resolveGatewayModel,
        });

        attempts.push(evaluation.attempt as CallRetryAttempt<MODEL>);

        /**
         * No retry matched. Surface the error, wrapped in a `RetryError` when
         * more than one attempt was made.
         */
        if (!evaluation.retryModel) {
          recorder?.endAttempt({
            attempt: attemptNumber,
            outcome: 'failure',
            error,
          });
          throw evaluation.finalError;
        }

        /**
         * The caller has cancelled. Any re-run would forward that dead signal
         * and abort instantly, so respect the cancel rather than fire a doomed
         * retry.
         */
        if (callerSignal?.aborted) {
          recorder?.endAttempt({
            attempt: attemptNumber,
            outcome: 'failure',
            error,
          });
          throw error;
        }

        const retryModel = evaluation.retryModel as unknown as Retry<
          MODEL,
          INPUT
        >;
        const backoff = resolveBackoffDelay(retryModel, attempts);

        recorder?.endAttempt({
          attempt: attemptNumber,
          outcome: 'retry',
          error,
          delayMs: backoff,
        });

        if (backoff !== undefined) {
          await delay(backoff, { abortSignal: callerSignal });
        }

        currentModel = retryModel.model;
        currentRetry = retryModel;
        continue;
      }

      /**
       * The attempt produced an outcome the caller has not seen yet, so result
       * conditions still get a say and fail-over is still possible.
       */
      if (settled.type === 'result') {
        const finishReason = finishReasonOf(settled.result);

        const resultAttempt: CallRetryResultAttempt<MODEL> = {
          type: 'result',
          result: settled.result,
          model: attemptModel,
          options: attemptArgs as unknown as CallArgs<MODEL>,
        };

        const context = {
          current: resultAttempt,
          attempts: [...attempts, resultAttempt],
        } as unknown as CallRetryContext<MODEL>;

        const retryModel = (await findRetryModel<MODEL, INPUT>(
          options.retries,
          context as never,
          entryPoint.resolveGatewayModel,
        )) as unknown as Retry<MODEL, INPUT> | undefined;

        if (retryModel) {
          attempts.push(resultAttempt);

          const backoff = resolveBackoffDelay(retryModel, attempts);

          recorder?.endAttempt({
            attempt: attemptNumber,
            outcome: 'retry',
            finishReason,
            delayMs: backoff,
          });

          if (backoff !== undefined) {
            await delay(backoff, { abortSignal: callerSignal });
          }

          currentModel = retryModel.model;
          currentRetry = retryModel;
          continue;
        }

        recorder?.endAttempt({
          attempt: attemptNumber,
          outcome: 'success',
          finishReason,
        });
        options.onSuccess?.({
          current: {
            type: 'success',
            model: attemptModel,
            result,
            finishReason,
          },
          attempts: [...attempts],
        });
        return result;
      }

      recorder?.endAttempt({ attempt: attemptNumber, outcome: 'success' });
      options.onSuccess?.({
        current: { type: 'success', model: attemptModel, result },
        attempts: [...attempts],
      });
      return result;
    }
  } catch (error) {
    /**
     * Every way the loop can end without producing a result lands here: no
     * retry matched, the caller's signal was already aborted, the caller
     * aborted during a backoff delay, or a caller-supplied handler threw.
     * Reporting once at the boundary is what makes it impossible to reject
     * without telling `onFailure` and the operation span about it.
     */
    operationError = error;
    emitFailure(options, attempts, error);
    throw error;
  } finally {
    recorder?.endOperation({
      provider: currentModel.provider,
      modelId: currentModel.modelId,
      error: operationError,
    });
  }
}
