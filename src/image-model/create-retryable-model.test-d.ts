import { assertType, describe, expectTypeOf, it } from 'vitest';
import { MockImageModel } from '../internal/test-utils.js';
import type {
  ImageModel,
  ImageModelCallOptions,
  ImageModelGenerate,
  SuccessContext,
} from '../types.js';
import { createRetryableModel } from './create-retryable-model.js';

describe('createRetryableModel', () => {
  it('should return ImageModel for a model instance', () => {
    const retryable = createRetryableModel({
      model: new MockImageModel(),
      retries: [new MockImageModel(), { model: new MockImageModel() }],
    });

    assertType<ImageModel>(retryable);
    expectTypeOf(retryable).toEqualTypeOf<ImageModel>();
  });

  it('should return ImageModel for a gateway string', () => {
    const retryable = createRetryableModel({
      model: 'google/imagen-4.0-generate-001',
      retries: ['google/imagen-4.0-fast-generate-001'],
    });

    assertType<ImageModel>(retryable);
    expectTypeOf(retryable).toEqualTypeOf<ImageModel>();
  });

  it('should type SuccessContext correctly', () => {
    type Ctx = SuccessContext<ImageModel>;

    expectTypeOf<Ctx['current']['model']>().toEqualTypeOf<ImageModel>();
    expectTypeOf<
      Ctx['current']['result']
    >().toEqualTypeOf<ImageModelGenerate>();
    expectTypeOf<
      Ctx['current']['options']
    >().toEqualTypeOf<ImageModelCallOptions>();
  });
});
