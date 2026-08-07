import { generateText, type ToolSet } from 'ai';
import { resolveLanguageModel } from '../../../internal/resolve-model.js';
import type { LanguageModel } from '../../../types.js';
import type { GenerateTextInput } from '../../inputs.js';
import type { CallRetryArg } from '../../retry-arg.js';
import { defineRetryableCall, viaTimeoutArg } from '../../retryable-calls.js';
import { tagResult } from '../../tag-result.js';

/**
 * `generateText` with call-level retries.
 *
 * Takes exactly the arguments `generateText` takes, plus `retry`. The model
 * stays a normal argument and is swapped per attempt, and each attempt is a
 * fresh call — which is what makes a call-level deadline (`timeout`) or an
 * inbound cancellation recoverable at all. A retry running *below* the model
 * cannot see either: by the time one fires, the SDK has already torn the call
 * down and discarded whatever the lower retry produced.
 *
 * @example
 * const result = await retryableGenerateText({
 *   model: openai('gpt-4o'),
 *   prompt: 'Invent a new holiday.',
 *   timeout: { totalMs: 5_000 },
 *   retry: [serviceOverloaded(fallbackModel)],
 * });
 */
export type RetryableGenerateText = <
  TOOLS extends ToolSet,
  INPUT extends GenerateTextInput = GenerateTextInput,
>(
  args: Omit<Parameters<typeof generateText>[0], 'tools' | 'activeTools'> & {
    tools?: TOOLS;
    activeTools?: Array<keyof TOOLS & string>;
    retry?: CallRetryArg<
      LanguageModel,
      INPUT,
      GenerateTextInput,
      Awaited<ReturnType<typeof generateText<TOOLS>>>
    >;
  },
) => ReturnType<typeof generateText<TOOLS>>;

export const retryableGenerateText = defineRetryableCall<
  LanguageModel,
  Parameters<typeof generateText>[0],
  Awaited<ReturnType<typeof generateText>>
>({
  operation: 'generateText',
  genAiOperation: 'chat',
  resolveGatewayModel: resolveLanguageModel,
  call: generateText,
  deadline: viaTimeoutArg,
  /**
   * A resolved `generateText` is a complete generation, so it is always worth
   * judging against result conditions before it is handed over.
   */
  settle: async (result) => ({
    type: 'result',
    result: tagResult('generateText', result),
  }),
}) as RetryableGenerateText;
