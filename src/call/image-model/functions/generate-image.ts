import { generateImage } from 'ai';
import { resolveImageModel } from '../../../internal/resolve-model.js';
import type { ImageModel } from '../../../types.js';
import type { GenerateImageInput } from '../../inputs.js';
import type { CallRetryArg } from '../../retry-arg.js';
import { defineRetryableCall, viaAbortSignal } from '../../retryable-calls.js';
import { tagResult } from '../../tag-result.js';

/**
 * `generateImage` with call-level retries.
 *
 * Takes exactly the arguments `generateImage` takes, plus `retry`. Having no
 * `timeout` argument of its own, a `Retry.timeout` here is composed into
 * `abortSignal` alongside the caller's own signal.
 */
export type RetryableGenerateImage = <
  INPUT extends GenerateImageInput = GenerateImageInput,
>(
  args: Parameters<typeof generateImage>[0] & {
    retry?: CallRetryArg<
      ImageModel,
      INPUT,
      GenerateImageInput,
      Awaited<ReturnType<typeof generateImage>>
    >;
  },
) => ReturnType<typeof generateImage>;

export const retryableGenerateImage = defineRetryableCall<
  ImageModel,
  Parameters<typeof generateImage>[0],
  Awaited<ReturnType<typeof generateImage>>
>({
  operation: 'generateImage',
  genAiOperation: 'generate_content',
  resolveGatewayModel: resolveImageModel,
  call: generateImage,
  deadline: viaAbortSignal,
  /**
   * A resolved `generateImage` is a complete result, so it is always worth
   * judging against result conditions before it is handed over.
   */
  settle: async (result) => ({
    type: 'result',
    result: tagResult('generateImage', result),
  }),
}) as RetryableGenerateImage;
