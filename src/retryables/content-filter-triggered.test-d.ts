import { describe, expectTypeOf, it } from 'vitest';
import {
  createRetryableModel,
  MockLanguageModel,
} from '../internal/test-utils.js';
import type { Retryable } from '../types.js';
import { contentFilterTriggered } from './content-filter-triggered.js';

describe('contentFilterTriggered', () => {
  it('should accept language model instance', () => {
    const model = MockLanguageModel.from();
    const retryable = contentFilterTriggered(model);

    const retryableModel = createRetryableModel({
      model,
      retries: [retryable],
    });

    expectTypeOf(retryable).toEqualTypeOf<Retryable<MockLanguageModel>>();
  });

  it('should accept language model with options', () => {
    const model = MockLanguageModel.from();
    const retryable = contentFilterTriggered(model, { maxAttempts: 3 });

    const retryableModel = createRetryableModel({
      model,
      retries: [retryable],
    });

    expectTypeOf(retryable).toEqualTypeOf<Retryable<MockLanguageModel>>();
  });

  it('should accept model string', () => {
    const retryable = contentFilterTriggered('openai/gpt-4.1');

    const retryableModel = createRetryableModel({
      model: 'openai/gpt-4.1',
      retries: [retryable],
    });

    // expectTypeOf(retryable).toEqualTypeOf<Retryable<LanguageModel>>();
  });
});
