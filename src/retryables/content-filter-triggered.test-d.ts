import { describe, expectTypeOf, it } from 'vitest';
import { createRetryable } from '../create-retryable-model.js';
import { MockLanguageModel } from '../test-utils.js';
import type { Retryable } from '../types.js';
import { contentFilterTriggered } from './content-filter-triggered.js';

describe('contentFilterTriggered', () => {
  it('should accept language model instance', () => {
    const model = new MockLanguageModel();
    const retryable = contentFilterTriggered(model);

    const retryableModel = createRetryable({
      model,
      retries: [retryable],
    });

    expectTypeOf(retryable).toEqualTypeOf<Retryable<MockLanguageModel>>();
  });

  it('should accept language model with options', () => {
    const model = new MockLanguageModel();
    const retryable = contentFilterTriggered(model, { maxAttempts: 3 });

    const retryableModel = createRetryable({
      model,
      retries: [retryable],
    });

    expectTypeOf(retryable).toEqualTypeOf<Retryable<MockLanguageModel>>();
  });

  it('should accept model string', () => {
    const retryable = contentFilterTriggered('openai/gpt-4.1');

    const retryableModel = createRetryable({
      model: 'openai/gpt-4.1',
      retries: [retryable],
    });

    // expectTypeOf(retryable).toEqualTypeOf<Retryable<LanguageModel>>();
  });
});
