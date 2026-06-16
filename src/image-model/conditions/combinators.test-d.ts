import { google } from '@ai-sdk/google';
import { openai } from '@ai-sdk/openai';
import { describe, expectTypeOf, it } from 'vitest';
import type { ResolvableImageModel, Retryable } from '../../types.js';
import { and, error, httpStatus, noImage, not, or } from '../index.js';

describe('top-level combinators (image-model)', () => {
  it('infer the image family and finalize without casts', () => {
    expectTypeOf(
      or(httpStatus(529), noImage()).switch({
        model: google.image('gemini-3-pro-image-preview'),
      }),
    ).toMatchTypeOf<Retryable<ResolvableImageModel>>();

    expectTypeOf(
      and(httpStatus(503), not(error.isRetryable(false))).switch({
        model: openai.image('dall-e-3'),
      }),
    ).toMatchTypeOf<Retryable<ResolvableImageModel>>();

    // gateway image string is accepted as a switch target
    expectTypeOf(
      noImage().switch({ model: 'google/imagen-4.0-generate-001' }),
    ).toMatchTypeOf<Retryable<ResolvableImageModel>>();
  });
});
