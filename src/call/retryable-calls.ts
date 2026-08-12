import type { TimeoutConfiguration, ToolSet } from 'ai';
import type { GatewayResolver } from '../internal/resolve-model.js';
import {
  mergeRetryTimeout,
  totalTimeoutMs,
} from '../internal/retry-timeout.js';
import type { GenAiOperation } from '../internal/telemetry.js';
import type { AnyModel, TotalTimeout } from '../types.js';
import { type CallRetryArg, toCallRetryOptions } from './retry-arg.js';
import {
  type RetryLoopArgs,
  type DeadlineStrategy,
  type EntryPoint,
  runRetryLoop,
  type Settled,
} from './run-retry-loop.js';

/**
 * The machinery every call-level entry point is built from. One file per entry
 * point sits alongside this one, each declaring its own row, its own signature
 * and its own export, so everything specific to `streamText` — how it takes a
 * deadline, how its outcome is decided, what it returns — reads in one place.
 *
 * Only the two deadline strategies are genuinely shared, and they are what the
 * five entry points actually differ by: `generateText` and `streamText` have a
 * `timeout` argument, `embed`, `embedMany` and `generateImage` have nothing but
 * `abortSignal`.
 */

/** Arguments of an entry point that carries its own `timeout` configuration. */
type TimeoutArgs = { timeout?: TimeoutConfiguration<ToolSet> };

/**
 * Deadline strategy for entry points that take a `timeout` argument.
 *
 * Preferred wherever it exists: the SDK enforces it around the whole call and
 * reports it as an `abort` part, which is exactly what a call-level retry needs
 * to see. Composing into `abortSignal` instead would be wrong here, because an
 * inbound signal is deliberately a hard caller-cancel that must *not* fail over.
 *
 * The retry's deadline is merged into the call's rather than replacing it, so
 * one that narrows `totalMs` leaves a `firstChunkMs` the caller configured
 * standing.
 */
export const viaTimeoutArg: DeadlineStrategy<TimeoutArgs> = (args, timeout) =>
  timeout === undefined
    ? args
    : { ...args, timeout: mergeRetryTimeout(args.timeout, timeout) };

/**
 * Arguments of an entry point that has no `timeout` of its own, so this library
 * lends it one. The SDK function never sees the key: it is read here and turned
 * into a signal, then dropped.
 */
type SignalOnlyArgs = RetryLoopArgs & { timeout?: TotalTimeout };

/**
 * Deadline strategy for entry points that have no `timeout` argument at all —
 * only `abortSignal`. The caller's signal is composed in rather than replaced,
 * so a genuine cancel still propagates mid-attempt.
 *
 * The deadline can come from the retry or from the call itself, the retry
 * winning where both name one. Either way a *fresh* signal is built per
 * attempt, which is the whole reason the call's own `timeout` is worth
 * lending: a deadline the caller bakes into `abortSignal` instead is a hard
 * cancel that deliberately does not fail over, so it could only ever kill the
 * first attempt and every retry with it. The budget is per attempt rather than
 * a ceiling on the loop, matching what the entry points with a real `timeout`
 * argument do.
 *
 * Only a total budget survives the trip through a signal; see
 * {@link totalTimeoutMs}.
 */
export const viaAbortSignal: DeadlineStrategy<SignalOnlyArgs> = (
  args,
  timeout,
  callerSignal,
) => {
  const { timeout: callTimeout, ...callArgs } = args;
  const timeoutMs = totalTimeoutMs(timeout) ?? totalTimeoutMs(callTimeout);
  if (timeoutMs === undefined) return callArgs as SignalOnlyArgs;

  const deadline = AbortSignal.timeout(timeoutMs);
  return {
    ...callArgs,
    abortSignal: callerSignal
      ? AbortSignal.any([callerSignal, deadline])
      : deadline,
  } as SignalOnlyArgs;
};

/**
 * Build the public function for one entry point.
 *
 * Every one of them is this: describe how the entry point is called and how its
 * outcome is decided, then split `retry` off the arguments and hand the rest to
 * the loop.
 *
 * The row is written against the entry point's **real** argument and result
 * types, so `call` and `settle` are checked against what the SDK actually
 * produces, while the loop — generic over neither — only ever sees the erased
 * argument shape. That reconciliation is the single cast here, and the reason
 * the caller must finish with `as RetryableXxx`: the implementation cannot wear
 * a polymorphic signature, so the file that owns the entry point owns its type.
 */
export function defineRetryableCall<
  MODEL extends AnyModel,
  ARGS,
  RESULT,
  COMMIT = RESULT,
>(entry: {
  /** Span name and `ai_retry.operation` attribute. */
  operation: string;
  /** Standard `gen_ai.operation.name` value for the underlying model call. */
  genAiOperation: GenAiOperation;
  /** Resolves gateway model-id strings for this entry point's model family. */
  resolveGatewayModel: GatewayResolver;
  /** Issues one attempt. */
  call: (args: ARGS) => Promise<RESULT>;
  /** Applies the per-attempt deadline. */
  deadline: DeadlineStrategy<any>;
  /**
   * Decides whether a returned result is terminal or still judgeable against
   * result conditions, reporting it as the entry point's `COMMIT`. Omit where a
   * returned result is always terminal.
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
   * Hold the success report until the outcome is final. Only streaming needs
   * it; omit it and the loop reports as soon as it has a result.
   */
  deferSuccess?: (result: RESULT, report: () => void) => void;
}) {
  const entryPoint = entry as unknown as EntryPoint<
    MODEL,
    RetryLoopArgs,
    RESULT,
    COMMIT
  >;

  return (
    args: RetryLoopArgs & {
      retry?: CallRetryArg<MODEL, unknown, unknown, RESULT, COMMIT>;
    },
  ): Promise<RESULT> => {
    const { retry, ...callArgs } = args;
    return runRetryLoop({
      entryPoint,
      args: callArgs as RetryLoopArgs,
      options: toCallRetryOptions(retry),
    });
  };
}
