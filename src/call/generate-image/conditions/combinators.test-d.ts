import { openai } from '@ai-sdk/openai';
import { describe, expectTypeOf, it } from 'vitest';
import type { CallRetryable } from '../../types.js';
import type { ResolvableImageModel } from '../../../types.js';
import type { GenerateImageCommitResult } from '../types.js';
import { and, error, httpStatus, noImage, not, or, result } from './index.js';

const fallback = openai.image('dall-e-3');

describe('top-level combinators (call/generate-image)', () => {
  it('or/and/not infer the family and finalize to a call retryable', () => {
    expectTypeOf(
      or(httpStatus(529), noImage()).switch({ model: fallback }),
    ).toEqualTypeOf<CallRetryable<ResolvableImageModel>>();

    expectTypeOf(
      and(httpStatus(503), error.message('temporary')).switch({
        model: 'google/imagen-4.0-generate-001',
      }),
    ).toEqualTypeOf<CallRetryable<ResolvableImageModel>>();

    expectTypeOf(not(noImage()).retry({ maxAttempts: 2 })).toEqualTypeOf<
      CallRetryable<ResolvableImageModel>
    >();
  });

  it('carries the entry point commit result into a result retryable', () => {
    expectTypeOf(
      result((res) => res.images.length < 2).switch({ model: fallback }),
    ).toEqualTypeOf<
      CallRetryable<ResolvableImageModel, never, GenerateImageCommitResult>
    >();
  });

  it('rejects a fallback from a different family', () => {
    // @ts-expect-error a language model cannot answer an image call
    httpStatus(429).switch({ model: openai('gpt-4o') });
  });
});
