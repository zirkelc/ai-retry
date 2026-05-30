import { anthropic } from '@ai-sdk/anthropic';
import { openai } from '@ai-sdk/openai';
import { APICallError } from 'ai';
import { describe, expectTypeOf, it } from 'vitest';
import type { ResolvableLanguageModel, Retryable } from '../../types.js';
import { and, error, httpStatus, not, or } from '../index.js';

describe('top-level combinators (language-model)', () => {
  it('or/and/not infer the family and finalize without casts', () => {
    // or() with mixed low- and high-level conditions
    expectTypeOf(
      or(
        error((e) => APICallError.isInstance(e) && e.statusCode === 418),
        httpStatus(529, 'overloaded'),
      ).switch({ model: anthropic('claude-sonnet-4-0') }),
    ).toMatchTypeOf<Retryable<ResolvableLanguageModel>>();

    // and() finalized with a gateway string fallback
    expectTypeOf(
      and(httpStatus(503), error.message('temporary')).switch({
        model: 'openai/gpt-5',
      }),
    ).toMatchTypeOf<Retryable<ResolvableLanguageModel>>();

    // not() finalized with retry()
    expectTypeOf(
      not(error.isRetryable(true)).retry({ delay: 1_000, maxAttempts: 2 }),
    ).toMatchTypeOf<Retryable<ResolvableLanguageModel>>();
  });

  it('rejects an invalid switch model (inference is real, not any)', () => {
    // @ts-expect-error number is not a valid model
    or(httpStatus(429)).switch({ model: 123 });

    // accepts both a model instance and a gateway string literal
    or(httpStatus(429)).switch({ model: openai('gpt-4o') });
    or(httpStatus(429)).switch({ model: 'openai/gpt-5' });
  });
});
