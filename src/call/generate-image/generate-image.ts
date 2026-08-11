import { generateImage } from 'ai';
import { resolveImageModel } from '../../internal/resolve-model.js';
import type { ImageModel, TotalTimeout } from '../../types.js';
import type { GenerateImageInput } from '../inputs.js';
import type { CallRetryArg } from '../retry-arg.js';
import { defineRetryableCall, viaAbortSignal } from '../retryable-calls.js';

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
    /**
     * Deadline for each attempt, in milliseconds.
     *
     * `generateImage` has no timeout of its own; this one is this library's,
     * turned into a fresh `AbortSignal` per attempt and never passed on.
     * That freshness is the point: a deadline of your own, composed into
     * `abortSignal`, reads as a cancellation and stops the retry loop dead
     * rather than failing over.
     */
    timeout?: TotalTimeout;
    retry?: CallRetryArg<
      ImageModel,
      INPUT,
      GenerateImageInput,
      Awaited<ReturnType<typeof generateImage>>,
      Awaited<ReturnType<typeof generateImage>>,
      TotalTimeout
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
    result,
  }),
}) as RetryableGenerateImage;
