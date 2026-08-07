import type { ToolSet } from 'ai';
import { Condition } from '../../internal/conditions/condition.js';
import { isResultAttempt } from '../../internal/guards.js';
import type {
  AnyResolvableModel,
  ResolvableLanguageModel,
  ResolvedModel,
  ModelRetryAttempt,
} from '../../types.js';
import type { CallRetryContext, CallRetryResultAttempt } from '../types.js';
import type {
  CallFinishReason,
  CallLanguageModelResult,
  CallResult,
} from '../types.js';

/**
 * Read the result off an attempt that produced one.
 *
 * The attempt shape is identical across families here; only the result type
 * differs, and that is the caller's to name.
 */
function resultOf(attempt: ModelRetryAttempt<any>): unknown {
  return (attempt as unknown as CallRetryResultAttempt<any>).result;
}

/**
 * Build the result-side condition helper for the call layer, bound to one model
 * family.
 *
 * Unlike the model-level equivalent, the result handed to a predicate is the
 * entry point's own — the object the caller would have received — not a
 * provider result reconstructed from it. Which entry point produced it is
 * carried on the result itself, so a family reachable through more than one
 * (embedding via `embed`/`embedMany`) needs a guard before reading anything
 * specific to one of them.
 */
export function createCallResultAPI<BOUND extends AnyResolvableModel>() {
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
      res: CallResult<ResolvedModel<MODEL>>,
      ctx: CallRetryContext<MODEL>,
    ) => boolean | Promise<boolean>,
  ): Condition<MODEL, 'call'> {
    return new Condition<MODEL, 'call'>(async (ctx) => {
      const current = ctx.current as ModelRetryAttempt<any>;
      if (!isResultAttempt(current)) return false;
      return predicate(
        resultOf(current) as CallResult<ResolvedModel<MODEL>>,
        ctx,
      );
    });
  }

  return { result };
}

/**
 * The call-layer result API for language models.
 *
 * Differs from the generic one in two ways: `result` leads with a `TOOLS`
 * parameter, and `finishReason` is exposed as its own condition.
 */
export function createCallLanguageModelResultAPI<
  BOUND extends ResolvableLanguageModel,
>() {
  /**
   * Build a condition from a predicate over the current result. The predicate
   * runs only when the current attempt produced one; error attempts return
   * false.
   *
   * For a stream, "produced one" means it ended without emitting any content —
   * past the first content part the attempt is the caller's and can no longer
   * fail over. A pre-commit stream has no text and no tool calls by definition,
   * so conditions there are effectively finish-reason-shaped. That ceiling is
   * inherent to streaming, and the streaming member of the result union says so
   * by declaring no content at all.
   *
   * `TOOLS` names the tool set the tool calls should be typed against. It has to
   * be given here rather than at a guard, because narrowing works within the
   * type the predicate was handed. It is unchecked — nothing verifies it matches
   * the tools the call was issued with, the same contract as a cast.
   *
   * **Important:** returns a `Condition`, not a retryable. Call `.switch()` or
   * `.retry()` to plug it into `retry: [...]`.
   *
   * @example
   * result((res) => res.finishReason === 'length').switch({ model: fallback })
   *
   * @example
   * result<typeof tools>((res) =>
   *   isGenerateTextResult(res) && res.toolCalls.length === 0,
   * ).retry({ maxAttempts: 3 })
   */
  function result<TOOLS extends ToolSet = ToolSet, MODEL extends BOUND = BOUND>(
    predicate: (
      res: CallLanguageModelResult<TOOLS>,
      ctx: CallRetryContext<MODEL>,
    ) => boolean | Promise<boolean>,
  ): Condition<MODEL, 'call'> {
    return new Condition<MODEL, 'call'>(async (ctx) => {
      const current = ctx.current as ModelRetryAttempt<any>;
      if (!isResultAttempt(current)) return false;
      return predicate(
        resultOf(current) as CallLanguageModelResult<TOOLS>,
        ctx,
      );
    });
  }

  /**
   * Match the result's finish reason against one of the given values.
   *
   * Readable without a guard: both `generateText` and a contentless
   * `streamText` report it, in the same flat form the SDK uses.
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
  ): Condition<MODEL, 'call'> {
    return result<ToolSet, MODEL>((res) => reasons.includes(res.finishReason));
  }

  return { result: Object.assign(result, { finishReason }), finishReason };
}
