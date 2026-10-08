import { AISDKError } from 'ai';
import { Condition } from '../../internal/conditions/condition.js';
import { isErrorAttempt, isResultAttempt } from '../../internal/guards.js';
import type { AnyResolvableModel, ModelRetryAttempt } from '../../types.js';
import type { CallFinishReason, CallRetryResultAttempt } from '../types.js';

/**
 * The finish-reason condition of the call layer, and where an attempt's finish
 * reason is read from. The reading is shared with the retry loop's telemetry:
 * telemetry records what was reported, and the condition also decides which of
 * it counts.
 */

/**
 * The finish reason a result carries. Results of the language entry points
 * have one; embeddings and images do not, and report nothing rather than a
 * placeholder.
 */
export function resultFinishReason(
  result: unknown,
): CallFinishReason | undefined {
  return readFinishReason(result);
}

/**
 * The finish reason an error carries.
 *
 * The SDK keeps it on an error when it rejected a generation that had already
 * completed: a structured output that cannot be parsed, or a response without
 * a required tool call. A content filter answering with a plain-text refusal
 * arrives this way. Only SDK errors are read, so a foreign error that happens
 * to have such a field is not taken for one.
 */
export function errorFinishReason(
  error: unknown,
): CallFinishReason | undefined {
  return AISDKError.isInstance(error) ? readFinishReason(error) : undefined;
}

function readFinishReason(carrier: unknown): CallFinishReason | undefined {
  const reason = (carrier as { finishReason?: unknown } | undefined)
    ?.finishReason;
  return typeof reason === 'string' ? (reason as CallFinishReason) : undefined;
}

/**
 * The finish reasons that can explain why an error attempt failed.
 *
 * An error carrying `stop` or `tool-calls` comes from a generation that
 * finished normally and whose content was then rejected, such as JSON that does
 * not fit the schema. Its finish reason says nothing about the failure, so it is
 * not counted. Were it counted, `finishReason('stop')` would fail over on a
 * schema mismatch, and `not(finishReason('stop'))` would stop matching an error
 * it matched before errors were read at all. Listed by inclusion, so a finish
 * reason added to the SDK later is not counted until it is known to explain a
 * failure.
 */
const FAILURE_FINISH_REASONS: ReadonlySet<CallFinishReason> = new Set([
  'content-filter',
  'length',
  'error',
  'other',
]);

/**
 * The finish reason of an attempt: off its result, or off its error when the
 * reason explains the failure.
 */
function attemptFinishReason(
  attempt: ModelRetryAttempt<any>,
): CallFinishReason | undefined {
  if (isResultAttempt(attempt)) {
    return resultFinishReason(
      (attempt as unknown as CallRetryResultAttempt<any, unknown>).result,
    );
  }
  if (isErrorAttempt(attempt)) {
    const reason = errorFinishReason(attempt.error);
    return reason !== undefined && FAILURE_FINISH_REASONS.has(reason)
      ? reason
      : undefined;
  }
  return undefined;
}

/**
 * The part of a commit result a finish-reason condition judges.
 */
export type FinishReasonCommitResult = { finishReason: CallFinishReason };

/**
 * Build the `finishReason` helper for an entry point whose commit result
 * reports one. Language entry points do; embeddings and images have no such
 * notion.
 *
 * The condition is typed against the finish reason alone, not the entry
 * point's whole commit result. One typed against the whole result would pin
 * the call's tools and output to their defaults, and a call passing either
 * would reject it.
 */
export function createFinishReasonAPI<BOUND extends AnyResolvableModel>() {
  /**
   * Match the finish reason against one of the given values.
   *
   * Also matches an attempt that failed with an SDK error carrying the finish
   * reason of the generation it rejected, such as a structured output that
   * cannot be parsed or a response without a required tool call. A content
   * filter answering with a plain-text refusal arrives this way. Only a reason
   * that can explain the failure counts there: `stop` and `tool-calls` on an
   * error never match.
   *
   * A streamed attempt is judged only up to its first content part: a refusal
   * that streams its text has committed by then, and only an error raised
   * before any content can be matched.
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
  ): Condition<MODEL, 'call', FinishReasonCommitResult> {
    return new Condition<MODEL, 'call', FinishReasonCommitResult>((ctx) => {
      const reason = attemptFinishReason(ctx.current as ModelRetryAttempt<any>);
      return reason !== undefined && reasons.includes(reason);
    });
  }

  return { finishReason };
}
