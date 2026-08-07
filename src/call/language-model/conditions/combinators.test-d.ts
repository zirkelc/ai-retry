import { anthropic } from '@ai-sdk/anthropic';
import { openai } from '@ai-sdk/openai';
import { APICallError } from 'ai';
import { describe, expectTypeOf, it } from 'vitest';
import type { CallRetryable } from '../../types.js';
import type { ResolvableLanguageModel } from '../../../types.js';
import {
  and,
  error,
  finishReason,
  httpStatus,
  not,
  or,
  result,
} from './index.js';

describe('top-level combinators (call/language-model)', () => {
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

  it('finalizes the result-side conditions to the same retryable', () => {
    expectTypeOf(
      finishReason('content-filter').switch({ model: openai('gpt-4o') }),
    ).toEqualTypeOf<CallRetryable<ResolvableLanguageModel>>();

    expectTypeOf(
      result((res) => res.finishReason === 'length').retry({ maxAttempts: 3 }),
    ).toEqualTypeOf<CallRetryable<ResolvableLanguageModel>>();
  });

  it('rejects an invalid switch model (inference is real, not any)', () => {
    // @ts-expect-error number is not a valid model
    or(httpStatus(429)).switch({ model: 123 });

    // accepts both a model instance and a gateway string literal
    or(httpStatus(429)).switch({ model: openai('gpt-4o') });
    or(httpStatus(429)).switch({ model: 'openai/gpt-5' });
  });

  it('rejects a fallback from a different family', () => {
    httpStatus(429).switch({
      // @ts-expect-error an embedding model cannot answer a language call
      model: openai.textEmbedding('text-embedding-3-small'),
    });
  });
});
