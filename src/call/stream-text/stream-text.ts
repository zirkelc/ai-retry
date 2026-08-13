import { streamText, type ToolSet } from 'ai';
import { detectStreamCommit } from './detect-stream-commit.js';
import type { CallRetryArg } from '../retry-arg.js';
import { resolveLanguageModel } from '../../internal/resolve-model.js';
import type { LanguageModel, StreamTimeout } from '../../types.js';
import type { StreamTextInput } from '../inputs.js';
import { defineRetryableCall, viaTimeoutArg } from '../retryable-calls.js';
import type { StreamTextCommitResult } from './types.js';

/**
 * The caller's stream callbacks, held until it is known whether this attempt is
 * the one they will hear about.
 *
 * Every attempt is issued with the caller's whole argument object, callbacks
 * included, so without this an attempt the loop goes on to discard still runs
 * their `onFinish`, `onAbort`, `onStepFinish` and `onError` — reporting a call
 * that completed, or an abort the caller never saw, for an attempt whose output
 * was thrown away. Those are exactly the callbacks side effects live in: usage
 * metrics, spans, trace writes.
 *
 * The caller cannot correct for it from outside. A discarded attempt's callback
 * necessarily fires *before* the loop has decided anything — deciding is what
 * it does next — so at that moment "discarded" and "about to fail terminally"
 * are the same event. Only the loop can tell them apart, and only afterwards.
 *
 * Holding costs nothing on the happy path: a committed stream's callbacks are
 * driven by the consumer reading it, which happens long after the loop settled,
 * so they are forwarded as they arrive. Only an attempt that finished before
 * the loop decided is ever queued, and only until it does.
 *
 * `onChunk` is held with the rest. A discarded attempt should never have fired
 * it — anything that would have committed the attempt — but the guarantee worth
 * offering is that a discarded attempt is silent, not that it is silent except
 * where we reason it cannot speak.
 */
type HeldCallbacks = {
  hold: <ARG>(callback: ((arg: ARG) => void) | undefined) => (arg: ARG) => void;
  release: () => void;
};

function holdCallbacks(): HeldCallbacks {
  let released = false;
  let queue: Array<() => void> = [];

  return {
    /**
     * Wraps one callback. Returns a handler even when the caller supplied
     * none, which is deliberate for `onError`: `streamText` defaults that to
     * `console.error`, and passing a handler unconditionally is what keeps
     * every discarded attempt out of the caller's logs.
     */
    hold<ARG>(callback: ((arg: ARG) => void) | undefined) {
      return (arg: ARG) => {
        if (callback === undefined) return;
        if (released) {
          callback(arg);
          return;
        }
        queue.push(() => callback(arg));
      };
    },
    release() {
      released = true;
      const pending = queue;
      queue = [];
      for (const run of pending) run();
    },
  };
}

/**
 * Held callbacks by the result they belong to. Weak because a discarded
 * attempt's result is dropped by the loop, and its queue — which will never be
 * released — should go with it.
 */
const heldByResult = new WeakMap<object, HeldCallbacks>();

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
  args: Parameters<typeof streamText<TOOLS>>[0] & {
    retry?: CallRetryArg<
      LanguageModel,
      INPUT,
      StreamTextInput,
      ReturnType<typeof streamText<TOOLS>>,
      StreamTextCommitResult,
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
    const held = holdCallbacks();
    const result = streamText({
      ...args,
      onChunk: held.hold(args.onChunk),
      onStepFinish: held.hold(args.onStepFinish),
      onFinish: held.hold(args.onFinish),
      onAbort: held.hold(args.onAbort),
      onError: held.hold(args.onError),
    });
    heldByResult.set(result, held);
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
   * This attempt is the one the caller gets, so everything its stream has said
   * so far was said on the caller's behalf after all.
   */
  release: (result) => heldByResult.get(result)?.release(),
}) as RetryableStreamText;
