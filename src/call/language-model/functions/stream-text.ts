import { streamText, type ToolSet } from 'ai';
import { detectStreamCommit } from '../../detect-stream-commit.js';
import { resolveLanguageModel } from '../../../internal/resolve-model.js';
import type { LanguageModel } from '../../../types.js';
import type { StreamTextInput } from '../../inputs.js';
import type { CallRetryArg } from '../../retry-arg.js';
import { defineRetryableCall, viaTimeoutArg } from '../../retryable-calls.js';

/**
 * `streamText` reports stream failures to `onError` rather than throwing, and
 * defaults it to `console.error` — which would log every attempt the loop went
 * on to recover from. A caller-supplied handler still wins.
 */
const IGNORE_STREAM_ERROR = () => {};

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
    retry?: CallRetryArg<
      LanguageModel,
      INPUT,
      StreamTextInput,
      ReturnType<typeof streamText<TOOLS>>
    >;
  },
) => Promise<ReturnType<typeof streamText<TOOLS>>>;

export const retryableStreamText = defineRetryableCall<
  LanguageModel,
  Parameters<typeof streamText>[0],
  ReturnType<typeof streamText>
>({
  operation: 'streamText',
  genAiOperation: 'chat',
  resolveGatewayModel: resolveLanguageModel,
  call: async (args) =>
    streamText({ ...args, onError: args.onError ?? IGNORE_STREAM_ERROR }),
  deadline: viaTimeoutArg,
  /**
   * `streamText` returns before anything has been generated, so the outcome
   * is read off the stream: committed at the first content part, judgeable if
   * it ends without one, thrown if it errors or trips a deadline first.
   */
  settle: (result, callerSignal) =>
    detectStreamCommit(result.stream, callerSignal),
}) as RetryableStreamText;
