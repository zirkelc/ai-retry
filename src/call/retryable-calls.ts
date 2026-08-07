import type { TimeoutConfiguration, ToolSet } from 'ai';
import type { GatewayResolver } from '../internal/resolve-model.js';
import type { GenAiOperation } from '../internal/telemetry.js';
import type { AnyModel } from '../types.js';
import { type CallRetryArg, toCallRetryOptions } from './retry-arg.js';
import type { CallResult } from './types.js';
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
 * Replace the total budget while preserving any finer-grained windows the
 * caller configured (`firstChunkMs`, `chunkMs`, `stepMs`, per-tool). A bare
 * number is shorthand for `totalMs`, so it is simply superseded.
 */
function mergeTimeout(
  base: TimeoutConfiguration<ToolSet> | undefined,
  totalMs: number,
): TimeoutConfiguration<ToolSet> {
  if (base === undefined || typeof base === 'number') return { totalMs };
  return { ...base, totalMs };
}

/**
 * Deadline strategy for entry points that take a `timeout` argument.
 *
 * Preferred wherever it exists: the SDK enforces it around the whole call and
 * reports it as an `abort` part, which is exactly what a call-level retry needs
 * to see. Composing into `abortSignal` instead would be wrong here, because an
 * inbound signal is deliberately a hard caller-cancel that must *not* fail over.
 */
export const viaTimeoutArg: DeadlineStrategy<TimeoutArgs> = (args, timeoutMs) =>
  timeoutMs === undefined
    ? args
    : { ...args, timeout: mergeTimeout(args.timeout, timeoutMs) };

/**
 * Deadline strategy for entry points that have no `timeout` argument at all —
 * only `abortSignal`. The caller's signal is composed in rather than replaced,
 * so a genuine cancel still propagates mid-attempt.
 */
export const viaAbortSignal: DeadlineStrategy<RetryLoopArgs> = (
  args,
  timeoutMs,
  callerSignal,
) => {
  if (timeoutMs === undefined) return args;
  const deadline = AbortSignal.timeout(timeoutMs);
  return {
    ...args,
    abortSignal: callerSignal
      ? AbortSignal.any([callerSignal, deadline])
      : deadline,
  };
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
   * result conditions. Omit where a returned result is always terminal.
   *
   * Throwing here is indistinguishable from the call throwing, which is what
   * lets a stream that fails before its first content part reuse the entire
   * error path with no branch in the loop.
   */
  settle?: (
    result: RESULT,
    callerSignal: AbortSignal | undefined,
  ) => Promise<Settled<CallResult<MODEL>>>;
}) {
  const entryPoint = entry as unknown as EntryPoint<
    MODEL,
    RetryLoopArgs,
    RESULT
  >;

  return (
    args: RetryLoopArgs & {
      retry?: CallRetryArg<MODEL, unknown, unknown, RESULT>;
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
