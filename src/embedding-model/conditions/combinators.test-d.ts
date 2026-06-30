import { openai } from '@ai-sdk/openai';
import { describe, expectTypeOf, it } from 'vitest';
import type { ResolvableEmbeddingModel, Retryable } from '../../types.js';
import { and, error, httpStatus, not, or } from '../index.js';

describe('top-level combinators (embedding-model)', () => {
  it('infer the embedding family and finalize without casts', () => {
    expectTypeOf(
      or(httpStatus(529), error.message('overloaded')).switch({
        model: openai.textEmbedding('text-embedding-3-small'),
      }),
    ).toMatchTypeOf<Retryable<ResolvableEmbeddingModel>>();

    expectTypeOf(
      and(httpStatus(503), not(error.isRetryable(false))).retry({
        delay: 1_000,
        maxAttempts: 2,
      }),
    ).toMatchTypeOf<Retryable<ResolvableEmbeddingModel>>();

    // gateway embedding string is accepted as a switch target
    expectTypeOf(
      httpStatus(529).switch({ model: 'openai/text-embedding-3-small' }),
    ).toMatchTypeOf<Retryable<ResolvableEmbeddingModel>>();
  });
});
