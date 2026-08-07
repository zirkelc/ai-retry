import { embed } from 'ai';
import { resolveEmbeddingModel } from '../../../internal/resolve-model.js';
import type { EmbeddingModel } from '../../../types.js';
import type { EmbedInput } from '../../inputs.js';
import type { CallRetryArg } from '../../retry-arg.js';
import { defineRetryableCall, viaAbortSignal } from '../../retryable-calls.js';
import { tagResult } from '../../tag-result.js';

/**
 * `embed` with call-level retries.
 *
 * Takes exactly the arguments `embed` takes, plus `retry`. Having no `timeout`
 * argument of its own, a `Retry.timeout` here is composed into `abortSignal`
 * alongside the caller's own signal.
 */
export type RetryableEmbed = <INPUT extends EmbedInput = EmbedInput>(
  args: Parameters<typeof embed>[0] & {
    retry?: CallRetryArg<
      EmbeddingModel,
      INPUT,
      EmbedInput,
      Awaited<ReturnType<typeof embed>>
    >;
  },
) => ReturnType<typeof embed>;

export const retryableEmbed = defineRetryableCall<
  EmbeddingModel,
  Parameters<typeof embed>[0],
  Awaited<ReturnType<typeof embed>>
>({
  operation: 'embed',
  genAiOperation: 'embeddings',
  resolveGatewayModel: resolveEmbeddingModel,
  call: embed,
  deadline: viaAbortSignal,
  /**
   * A resolved `embed` is a complete result, so it is always worth judging
   * against result conditions before it is handed over — an empty or degenerate
   * embedding is a fail-over case the error path never sees.
   */
  settle: async (result) => ({
    type: 'result',
    result: tagResult('embed', result),
  }),
}) as RetryableEmbed;
