import { describe, expectTypeOf, it } from 'vitest';
import {
  createRetryableModel,
  MockImageModel,
} from '../internal/test-utils.js';
import type { Retryable } from '../types.js';
import { noImageGenerated } from './no-image-generated.js';

describe('noImageGenerated', () => {
  it('should accept image model instance', () => {
    const model = MockImageModel.from();
    const retryable = noImageGenerated(model);

    const retryableModel = createRetryableModel({
      model,
      retries: [retryable],
    });

    expectTypeOf(retryable).toEqualTypeOf<Retryable<MockImageModel>>();
  });

  it('should accept image model with options', () => {
    const model = MockImageModel.from();
    const retryable = noImageGenerated(model, { maxAttempts: 3 });

    const retryableModel = createRetryableModel({
      model,
      retries: [retryable],
    });

    expectTypeOf(retryable).toEqualTypeOf<Retryable<MockImageModel>>();
  });
});
