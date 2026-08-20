import { anthropic } from '@ai-sdk/anthropic';
import { openai } from '@ai-sdk/openai';
import { APICallError } from 'ai';
import { describe, expectTypeOf, it } from 'vitest';
import type { ResolvableLanguageModel } from '../../../types.js';
import type { CallRetryable } from '../../types.js';
import type { StreamTextCommitResult } from '../types.js';
import {
  and,
  error,
  finishReason,
  httpStatus,
  not,
  or,
  result,
} from './index.js';

describe('top-level combinators (call/stream-text)', () => {
  it('or/and/not infer the family and finalize to a call retryable', () => {
    expectTypeOf(
      or(
        error((e) => APICallError.isInstance(e) && e.statusCode === 418),
        httpStatus(529, 'overloaded'),
      ).switch({ model: anthropic('claude-sonnet-4-0') }),
    ).toEqualTypeOf<CallRetryable<ResolvableLanguageModel>>();

    expectTypeOf(
      and(httpStatus(503), error.message('temporary')).switch({
        model: 'openai/gpt-5',
      }),
    ).toEqualTypeOf<CallRetryable<ResolvableLanguageModel>>();

    expectTypeOf(
      not(error.isRetryable(true)).retry({ delay: 1_000, maxAttempts: 2 }),
    ).toEqualTypeOf<CallRetryable<ResolvableLanguageModel>>();
  });

  it('carries the entry point commit result into the result-side retryables', () => {
    expectTypeOf(
      finishReason('content-filter').switch({ model: openai('gpt-4o') }),
    ).toEqualTypeOf<
      CallRetryable<ResolvableLanguageModel, never, StreamTextCommitResult>
    >();

    expectTypeOf(
      result((res) => res.finishReason === 'length').retry({ maxAttempts: 3 }),
    ).toEqualTypeOf<
      CallRetryable<ResolvableLanguageModel, never, StreamTextCommitResult>
    >();
  });

  it('rejects a fallback from a different family', () => {
    httpStatus(429).switch({
      // @ts-expect-error an embedding model cannot answer a language call
      model: openai.textEmbedding('text-embedding-3-small'),
    });
  });
});
