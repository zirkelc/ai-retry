import { Condition } from '../../internal/conditions/condition.js';
import { isResultAttempt } from '../../internal/guards.js';
import type { AnyResolvableModel, ModelRetryAttempt } from '../../types.js';
import type {
  CallFinishReason,
  CallRetryContext,
  CallRetryResultAttempt,
} from '../types.js';

/**
 * The result-side condition helpers of the call layer, shared by the entry
 * points that need no shaping of their own.
 *
 * Unlike the model-level equivalent, the result handed to a predicate is the
 * entry point's own — the object the caller would have received, or as much of
 * it as exists while failing over is still possible — not a provider result
 * reconstructed from it. Which entry point produced it is not carried on the
 * result and never has to be: an entry point's conditions are imported from its
 * own path and are typed against its commit result alone.
 */

/**
 * Read the result off an attempt that produced one.
 *
 * The attempt shape is identical across entry points here; only the result type
 * differs, and that is the caller's to name.
 */
function resultOf(attempt: ModelRetryAttempt<any>): unknown {
  return (attempt as unknown as CallRetryResultAttempt<any, unknown>).result;
}

/**
 * Evaluate a predicate against the current attempt's result, treating an error
 * attempt as no match.
 *
 * Exported for the entry points whose `result` cannot be an instantiation of
 * {@link createCallResultAPI} because their commit result carries a generic of
 * its own, and so have to declare the helper themselves.
 */
export function resultCondition<MODEL extends AnyResolvableModel, COMMIT>(
  predicate: (
    res: COMMIT,
    ctx: CallRetryContext<MODEL, COMMIT>,
  ) => boolean | Promise<boolean>,
): Condition<MODEL, 'call', COMMIT> {
  return new Condition<MODEL, 'call', COMMIT>(async (ctx) => {
    const current = ctx.current as ModelRetryAttempt<any>;
    if (!isResultAttempt(current)) return false;
    return predicate(resultOf(current) as COMMIT, ctx);
  });
}

/**
 * Build the result-side condition helper for one entry point, bound to its
 * model family and to the result it commits.
 */
export function createCallResultAPI<
  BOUND extends AnyResolvableModel,
  COMMIT,
>() {
  /**
   * Build a condition from a predicate over the current result. The predicate
   * runs only when the current attempt produced one; error attempts return
   * false.
   *
   * **Important:** returns a `Condition`, not a retryable. Call `.switch()` or
   * `.retry()` to plug it into `retry: [...]`.
   *
   * @example
   * result((res) => res.images.length < 2).switch({ model: fallback })
   */
  function result<MODEL extends BOUND = BOUND>(
    predicate: (
      res: COMMIT,
      ctx: CallRetryContext<MODEL, COMMIT>,
    ) => boolean | Promise<boolean>,
  ): Condition<MODEL, 'call', COMMIT> {
    return resultCondition<MODEL, COMMIT>(predicate);
  }

  return { result };
}

/**
 * Build the `finishReason` helper for an entry point whose commit result
 * reports one. Language entry points do; embeddings and images have no such
 * notion.
 */
export function createFinishReasonAPI<
  BOUND extends AnyResolvableModel,
  COMMIT extends { finishReason: CallFinishReason },
>() {
  /**
   * Match the result's finish reason against one of the given values.
   *
   * **Important:** returns a `Condition`, not a retryable. Call `.switch()` or
   * `.retry()` to plug it into `retry: [...]`.
   *
   * @example
   * finishReason('content-filter').switch({ model: fallback })
   * finishReason('length').retry({ maxAttempts: 3 })
   */
  function finishReason<MODEL extends BOUND = BOUND>(
    ...reasons: Array<CallFinishReason>
  ): Condition<MODEL, 'call', COMMIT> {
    return resultCondition<MODEL, COMMIT>((res) =>
      reasons.includes(res.finishReason),
    );
  }

  return { finishReason };
}
