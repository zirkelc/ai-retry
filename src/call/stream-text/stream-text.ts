import { streamText, type ToolSet } from 'ai';
import { detectStreamCommit } from './detect-stream-commit.js';
import { resolveLanguageModel } from '../../internal/resolve-model.js';
import type { LanguageModel, StreamTimeout } from '../../types.js';
import type { StreamTextInput } from '../inputs.js';
import { defineRetryableCall, viaTimeoutArg } from '../retryable-calls.js';
import type { StreamTextCommitResult, StreamTextRetryArg } from './types.js';

/**
 * `streamText` reports stream failures to `onError` rather than throwing, and
 * defaults it to `console.error` — which would log every attempt the loop went
 * on to recover from. A caller-supplied handler still wins.
 */
const IGNORE_STREAM_ERROR = () => {};

/**
 * What one attempt's stream has done so far, as far as reporting a clean end
 * is concerned.
 *
 * Three facts about `streamText` shape this, and each would report success for
 * a stream that had none:
 *
 * - `onFinish` fires after `onError` too. A stream carrying an `error` part
 *   still finishes, so `errored` is what separates ending from ending well.
 * - A stream can finish before the loop has decided, which is exactly what a
 *   contentless finish matching no condition does. `finished` outlives that
 *   race, so the order of the two stops mattering.
 * - Every attempt gets these handlers, and a losing one still finishes: the
 *   loop drains a contentless stream to judge it. Nothing marks the winner,
 *   because `report` already does — it is handed over only to the attempt the
 *   loop settled on, so a discarded one has nothing to call.
 *
 * A deadline needs no field at all: an aborted stream fires neither handler.
 */
type StreamOutcome = {
  errored: boolean;
  finished: boolean;
  report?: () => void;
};

/**
 * Attempts, by the result they produced. Weak because a losing attempt's
 * result is dropped by the loop and nothing here should keep it alive.
 */
const outcomes = new WeakMap<object, StreamOutcome>();

/** Report a clean end exactly once, whichever fact arrives last. */
function reportIfClean(outcome: StreamOutcome): void {
  if (!outcome.finished || outcome.errored) return;
  const report = outcome.report;
  outcome.report = undefined;
  report?.();
}

/**
 * `streamText` with call-level retries.
 *
 * Takes exactly the arguments `streamText` takes, plus `retry`. Fails over
 * while the attempt is still recoverable — an error, a `timeout` deadline, or a
 * finish with no content at all — and stops the moment the first content part
 * reaches the stream, after which the stream is the caller's.
 *
 * Returns a **promise** for the stream result, where `streamText` returns it
 * synchronously: the loop has to know which attempt won before it can hand
 * anything back. This is the only place the signature differs.
 *
 * @example
 * const result = await retryableStreamText({
 *   model: openai('gpt-4o'),
 *   prompt: 'Invent a new holiday.',
 *   timeout: { firstChunkMs: 2_000 },
 *   retry: [timeout().switch({ model: fastModel })],
 * });
 *
 * for await (const chunk of result.textStream) process.stdout.write(chunk);
 */
export type RetryableStreamText = <
  TOOLS extends ToolSet,
  INPUT extends StreamTextInput = StreamTextInput,
>(
  args: Omit<Parameters<typeof streamText>[0], 'tools' | 'activeTools'> & {
    tools?: TOOLS;
    activeTools?: Array<keyof TOOLS & string>;
    retry?: StreamTextRetryArg<
      LanguageModel,
      INPUT,
      StreamTextInput,
      ReturnType<typeof streamText<TOOLS>>,
      StreamTimeout
    >;
  },
) => Promise<ReturnType<typeof streamText<TOOLS>>>;

export const retryableStreamText = defineRetryableCall<
  LanguageModel,
  Parameters<typeof streamText>[0],
  ReturnType<typeof streamText>,
  StreamTextCommitResult
>({
  operation: 'streamText',
  genAiOperation: 'chat',
  resolveGatewayModel: resolveLanguageModel,
  call: async (args) => {
    const outcome: StreamOutcome = { errored: false, finished: false };
    const callerOnError = args.onError ?? IGNORE_STREAM_ERROR;
    const callerOnFinish = args.onFinish;

    const result = streamText({
      ...args,
      /** Composed, never replaced: the caller's handlers still run. */
      onError: (event) => {
        outcome.errored = true;
        callerOnError(event);
      },
      onFinish: (event) => {
        outcome.finished = true;
        callerOnFinish?.(event);
        reportIfClean(outcome);
      },
    });

    outcomes.set(result, outcome);
    return result;
  },
  deadline: viaTimeoutArg,
  /**
   * `streamText` returns before anything has been generated, so the outcome
   * is read off the stream: committed at the first content part, judgeable if
   * it ends without one, thrown if it errors or trips a deadline first.
   */
  settle: (result, callerSignal) =>
    detectStreamCommit(result.stream, callerSignal),
  /**
   * The loop has settled on this attempt, so its stream may now report a clean
   * end — which may already have happened, for a stream that finished without
   * content and matched no condition.
   */
  deferSuccess: (result, report) => {
    const outcome = outcomes.get(result);
    if (outcome === undefined) return;
    outcome.report = report;
    reportIfClean(outcome);
  },
}) as RetryableStreamText;
