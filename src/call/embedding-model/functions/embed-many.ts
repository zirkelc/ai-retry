import { embedMany } from 'ai';
import { resolveEmbeddingModel } from '../../../internal/resolve-model.js';
import type { EmbeddingModel } from '../../../types.js';
import type { EmbedManyInput } from '../../inputs.js';
import type { CallRetryArg } from '../../retry-arg.js';
import { defineRetryableCall, viaAbortSignal } from '../../retryable-calls.js';
import { tagResult } from '../../tag-result.js';

/**
 * `embedMany` with call-level retries.
 *
 * Takes exactly the arguments `embedMany` takes, plus `retry`. Having no
 * `timeout` argument of its own, a `Retry.timeout` here is composed into
 * `abortSignal` alongside the caller's own signal.
 *
 * A retry re-runs the whole call, so one failed batch re-embeds every value
 * rather than just the values in that batch.
 */
export type RetryableEmbedMany = <
  INPUT extends EmbedManyInput = EmbedManyInput,
>(
  args: Parameters<typeof embedMany>[0] & {
    retry?: CallRetryArg<
      EmbeddingModel,
      INPUT,
      EmbedManyInput,
      Awaited<ReturnType<typeof embedMany>>
    >;
  },
) => ReturnType<typeof embedMany>;

export const retryableEmbedMany = defineRetryableCall<
  EmbeddingModel,
  Parameters<typeof embedMany>[0],
  Awaited<ReturnType<typeof embedMany>>
>({
  operation: 'embedMany',
  genAiOperation: 'embeddings',
  resolveGatewayModel: resolveEmbeddingModel,
  call: embedMany,
  deadline: viaAbortSignal,
  /**
   * A resolved `embedMany` is a complete result, so it is always worth judging
   * against result conditions before it is handed over.
   */
  settle: async (result) => ({
    type: 'result',
    result: tagResult('embedMany', result),
  }),
}) as RetryableEmbedMany;
