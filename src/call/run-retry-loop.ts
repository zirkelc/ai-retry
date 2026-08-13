import { delay } from '@ai-sdk/provider-utils';
import { evaluateError } from '../internal/evaluate-error.js';
import { findRetryModel } from '../internal/find-retry-model.js';
import { resolveBackoffDelay } from '../internal/resolve-backoff-delay.js';
import { totalTimeoutMs } from '../internal/retry-timeout.js';
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
  RetryTimeout,
} from '../types.js';
import type {
  CallArgs,
  CallFinishReason,
  CallRetryAttempt,
  CallRetryContext,
  CallRetryResultAttempt,
} from './types.js';
import type {
  CallRetryOptions,
  CallSettledAttempt,
  CallSettledEvent,
} from './retry-arg.js';

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
  timeout: RetryTimeout | undefined,
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
  COMMIT = RESULT,
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
   * result conditions, and reports it in the shape conditions see. Omitted
   * where a returned result is always terminal.
   *
   * Throwing here is indistinguishable from the call throwing, which is what
   * lets a stream that fails before its first content part reuse the entire
   * error path with no branch in the loop.
   */
  settle?: (
    result: RESULT,
    callerSignal: AbortSignal | undefined,
  ) => Promise<Settled<COMMIT>>;
  /**
   * Tells the entry point which attempt the caller ends up with, success or
   * failure, once no further fail-over can change it.
   *
   * Every attempt is issued with the caller's own arguments, so an entry point
   * that hands those to something which calls back — a stream's `onFinish` —
   * can hold that back until this says the attempt is the caller's. Attempts
   * the loop discards are never named here, and stay silent.
   */
  release?: (result: RESULT) => void;
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
 * Report what the call amounted to, exactly once.
 *
 * `attempts` arrives holding every attempt that failed or was judged; a
 * success appends the attempt that ended the loop, which the loop otherwise
 * never records. Either way the array ends with the attempt that settled the
 * operation and its length is the attempt count — the same number the
 * operation span reports as `ai_retry.attempts`.
 */
function reportSettled<MODEL extends AnyModel, INPUT, OVERRIDE, RESULT, COMMIT>(
  options: CallRetryOptions<MODEL, INPUT, OVERRIDE, RESULT, COMMIT>,
  event: CallSettledEvent<MODEL, RESULT, COMMIT>,
): void {
  options.onSettled?.(event);
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
  COMMIT,
>(input: {
  entryPoint: EntryPoint<MODEL, ARGS, RESULT, COMMIT>;
  args: ARGS;
  options: CallRetryOptions<MODEL, INPUT, OVERRIDE, RESULT, COMMIT>;
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
   *
   * The deadline strategy still runs, with no retry deadline to apply. For an
   * entry point with a real `timeout` argument that changes nothing, and for
   * one whose `timeout` this library lends it, it is what keeps the argument
   * meaning the same thing either side of the switch.
   */
  if (isDisabled(options.disabled)) {
    return entryPoint.call(
      entryPoint.deadline(
        { ...args, model: baseModel },
        undefined,
        callerSignal,
      ),
    );
  }

  const recorder = await createRetryTelemetry(options.telemetry, {
    operation: entryPoint.operation,
    genAiOperation: entryPoint.genAiOperation,
    provider: baseModel.provider,
    modelId: baseModel.modelId,
  });

  const attempts: Array<CallRetryAttempt<MODEL, COMMIT>> = [];
  let currentModel = baseModel;
  let currentRetry: Retry<MODEL, INPUT> | undefined;

  let operationError: unknown;
  /**
   * The most recent attempt's result, kept so a terminal failure can still
   * release what that attempt held back. Undefined when the failing call never
   * returned one.
   */
  let lastResult: RESULT | undefined;
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
        } as unknown as CallRetryContext<MODEL, COMMIT>;

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
        timeoutMs: totalTimeoutMs(attemptTimeout),
      });

      /**
       * Only the attempt itself is guarded. Everything that runs once the
       * outcome is known stays outside, so a throwing hook cannot be mistaken
       * for a failed attempt and re-run a call that already succeeded.
       */
      let result: RESULT;
      let settled: Settled<COMMIT>;
      try {
        result = await entryPoint.call(attemptArgs);
        lastResult = result;
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

        attempts.push(evaluation.attempt as CallRetryAttempt<MODEL, COMMIT>);

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

        const resultAttempt: CallRetryResultAttempt<MODEL, COMMIT> = {
          type: 'result',
          result: settled.result,
          model: attemptModel,
          options: attemptArgs as unknown as CallArgs<MODEL>,
        };

        const context = {
          current: resultAttempt,
          attempts: [...attempts, resultAttempt],
        } as unknown as CallRetryContext<MODEL, COMMIT>;

        const retryModel = (await findRetryModel<MODEL, INPUT, COMMIT>(
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
        reportSettled(options, {
          outcome: 'success',
          model: attemptModel,
          attempts: [
            ...attempts,
            { type: 'success', model: attemptModel, result },
          ] as Array<CallSettledAttempt<MODEL, RESULT, COMMIT>>,
          result,
        });
        entryPoint.release?.(result);
        return result;
      }

      recorder?.endAttempt({ attempt: attemptNumber, outcome: 'success' });
      reportSettled(options, {
        outcome: 'success',
        model: attemptModel,
        attempts: [
          ...attempts,
          { type: 'success', model: attemptModel, result },
        ] as Array<CallSettledAttempt<MODEL, RESULT, COMMIT>>,
        result,
      });
      entryPoint.release?.(result);
      return result;
    }
  } catch (error) {
    /**
     * Every way the loop can end without producing a result lands here: no
     * retry matched, the caller's signal was already aborted, the caller
     * aborted during a backoff delay, or a caller-supplied handler threw.
     * Reporting once at the boundary is what makes it impossible to reject
     * without telling `onSettled` and the operation span about it.
     *
     * A rejection no attempt caused — one of your own callbacks throwing —
     * leaves `attempts` empty, and there is no call outcome to report.
     */
    operationError = error;
    if (attempts.length > 0) {
      reportSettled(options, {
        outcome: 'failure',
        model: currentModel,
        attempts: [...attempts] as Array<
          CallSettledAttempt<MODEL, RESULT, COMMIT>
        >,
        error,
      });
    }
    /**
     * The attempt that failed terminally is the caller's too: they receive its
     * error, so whatever it said belongs to them. Released after the report,
     * so `onSettled` is always the first word on the outcome.
     */
    if (lastResult !== undefined) entryPoint.release?.(lastResult);
    throw error;
  } finally {
    recorder?.endOperation({
      provider: currentModel.provider,
      modelId: currentModel.modelId,
      error: operationError,
    });
  }
}
