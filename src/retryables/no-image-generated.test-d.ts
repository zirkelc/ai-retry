import { describe, expectTypeOf, it } from 'vitest';
import { createRetryable } from '../create-retryable-model.js';
import { MockImageModel } from '../test-utils.js';
import type { Retryable } from '../types.js';
import { noImageGenerated } from './no-image-generated.js';

describe('noImageGenerated', () => {
  it('should accept image model instance', () => {
    const model = new MockImageModel();
    const retryable = noImageGenerated(model);

    const retryableModel = createRetryable({
      model,
      retries: [retryable],
    });

    expectTypeOf(retryable).toEqualTypeOf<Retryable<MockImageModel>>();
  });

  it('should accept image model with options', () => {
    const model = new MockImageModel();
    const retryable = noImageGenerated(model, { maxAttempts: 3 });

    const retryableModel = createRetryable({
      model,
      retries: [retryable],
    });

    expectTypeOf(retryable).toEqualTypeOf<Retryable<MockImageModel>>();
  });
});
