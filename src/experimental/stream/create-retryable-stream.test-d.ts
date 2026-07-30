import { describe, expectTypeOf, it } from 'vitest';
import { MockLanguageModel } from '../../internal/test-utils.js';
import type {
  LanguageModel,
  RetryAttempt,
  RetryCallOptions,
} from '../../types.js';
import { createRetryableStream } from './create-retryable-stream.js';

describe('createRetryableStream types', () => {
  it('should type the commit context with what the wrapper knows', () => {
    // Act
    createRetryableStream({
      model: MockLanguageModel.from(),
      retries: [],
      onCommit: (context) => {
        // Assert
        expectTypeOf(context.current.model).toEqualTypeOf<LanguageModel>();
        expectTypeOf(context.current.options).toEqualTypeOf<
          RetryCallOptions<LanguageModel>
        >();
        expectTypeOf(context.attempts).toEqualTypeOf<
          Array<RetryAttempt<LanguageModel>>
        >();
        expectTypeOf(context.current).not.toHaveProperty('result');
      },
    });
  });

  it('should not accept the call driver onComplete', () => {
    // Assert — the positive hook is named for this layer's boundary.
    expectTypeOf(createRetryableStream)
      .parameter(0)
      .not.toHaveProperty('onComplete');
  });
});
