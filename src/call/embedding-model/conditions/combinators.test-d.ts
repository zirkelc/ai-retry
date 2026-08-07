import { openai } from '@ai-sdk/openai';
import { describe, expectTypeOf, it } from 'vitest';
import type { CallRetryable } from '../../types.js';
import type { ResolvableEmbeddingModel } from '../../../types.js';
import {
  and,
  error,
  httpStatus,
  isEmbedResult,
  not,
  or,
  result,
} from './index.js';

const fallback = openai.textEmbedding('text-embedding-3-small');

describe('top-level combinators (call/embedding-model)', () => {
  it('or/and/not infer the family and finalize to a call retryable', () => {
    expectTypeOf(
      or(httpStatus(529), error.isRetryable(true)).switch({ model: fallback }),
    ).toEqualTypeOf<CallRetryable<ResolvableEmbeddingModel>>();

    expectTypeOf(
      and(httpStatus(503), error.message('temporary')).switch({
        model: 'openai/text-embedding-3-large',
      }),
    ).toEqualTypeOf<CallRetryable<ResolvableEmbeddingModel>>();

    expectTypeOf(
      not(error.isRetryable(false)).retry({ maxAttempts: 2 }),
    ).toEqualTypeOf<CallRetryable<ResolvableEmbeddingModel>>();
  });

  it('finalizes a result condition to the same retryable', () => {
    expectTypeOf(
      result((res) => isEmbedResult(res) && res.embedding.length === 0).switch({
        model: fallback,
      }),
    ).toEqualTypeOf<CallRetryable<ResolvableEmbeddingModel>>();
  });

  it('rejects a fallback from a different family', () => {
    // @ts-expect-error a language model cannot answer an embedding call
    httpStatus(429).switch({ model: openai('gpt-4o') });
  });
});
