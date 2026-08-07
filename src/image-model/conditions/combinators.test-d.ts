import { google } from '@ai-sdk/google';
import { openai } from '@ai-sdk/openai';
import { describe, expectTypeOf, it } from 'vitest';
import type { ResolvableImageModel, ModelRetryable } from '../../types.js';
import { and, error, httpStatus, noImage, not, or } from '../index.js';

describe('top-level combinators (image-model)', () => {
  it('infer the image family and finalize without casts', () => {
    expectTypeOf(
      or(httpStatus(529), noImage()).switch({
        model: google.image('gemini-3-pro-image-preview'),
      }),
    ).toEqualTypeOf<ModelRetryable<ResolvableImageModel, never>>();

    expectTypeOf(
      and(httpStatus(503), not(error.isRetryable(false))).switch({
        model: openai.image('dall-e-3'),
      }),
    ).toEqualTypeOf<ModelRetryable<ResolvableImageModel, never>>();

    // gateway image string is accepted as a switch target
    expectTypeOf(
      noImage().switch({ model: 'google/imagen-4.0-generate-001' }),
    ).toEqualTypeOf<ModelRetryable<ResolvableImageModel, never>>();
  });
});
