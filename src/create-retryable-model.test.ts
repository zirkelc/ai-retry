import { APICallError, generateText, RetryError } from 'ai';
import { describe, expect, it, vi } from 'vitest';
import {
  type CreateRetryableOptions,
  createRetryable,
  isErrorAttempt,
  isResultAttempt,
} from './create-retryable-model.js';
import { createMockModel } from './test-utils.js';
import type { LanguageModelV2Generate } from './types.js';

type OnError = Required<CreateRetryableOptions>['onError'];
type OnRetry = Required<CreateRetryableOptions>['onRetry'];

const mockResultText = 'Hello, world!';

const mockResult: LanguageModelV2Generate = {
  finishReason: 'stop',
  usage: { inputTokens: 10, outputTokens: 20, totalTokens: 30 },
  content: [{ type: 'text', text: mockResultText }],
  warnings: [],
};

const contentFilterResult: LanguageModelV2Generate = {
  finishReason: 'content-filter',
  usage: { inputTokens: 10, outputTokens: 20, totalTokens: 30 },
  content: [],
  warnings: [],
};

const retryableError = new APICallError({
  message: 'Rate limit exceeded',
  url: '',
  requestBodyValues: {},
  statusCode: 429,
  responseHeaders: {},
  responseBody:
    '{"error": {"message": "Rate limit exceeded", "code": "rate_limit_exceeded"}}',
  isRetryable: true,
  data: {
    error: {
      message: 'Rate limit exceeded',
      code: 'rate_limit_exceeded',
    },
  },
});

const nonRetryableError = new APICallError({
  message: 'Invalid API key',
  url: '',
  requestBodyValues: {},
  statusCode: 401,
  responseHeaders: {},
  responseBody:
    '{"error": {"message": "Invalid API key", "code": "invalid_api_key"}}',
  isRetryable: false,
  data: {
    error: {
      message: 'Invalid API key',
      code: 'invalid_api_key',
    },
  },
});

it('should succeed without errors', async () => {
  // Arrange
  const baseModel = createMockModel({
    finishReason: 'stop',
    usage: { inputTokens: 10, outputTokens: 20, totalTokens: 30 },
    content: [{ type: 'text', text: 'Hello, world!' }],
    warnings: [],
  });
  const retryableModel = createRetryable({
    model: baseModel,
    retries: [],
  });

  // Act
  const result = await generateText({
    model: retryableModel,
    prompt: 'Hello!',
  });

  // Assert
  expect(baseModel.doGenerateCalls.length).toBe(1);
  expect(result.text).toBe('Hello, world!');
});

describe('error-based retries', () => {
  it('should retry with errors', async () => {
    // Arrange
    const baseModel = createMockModel(retryableError);
    const fallbackModel = createMockModel(mockResult);

    // Act
    const result = await generateText({
      model: createRetryable({
        model: baseModel,
        retries: [
          (context) => {
            if (
              isErrorAttempt(context.current) &&
              APICallError.isInstance(context.current.error)
            ) {
              return { model: fallbackModel, maxAttempts: 1 };
            }
            return undefined;
          },
        ],
      }),
      prompt: 'Hello!',
      //
    });

    // Assert
    expect(baseModel.doGenerateCalls.length).toBe(1);
    expect(fallbackModel.doGenerateCalls.length).toBe(1);
    expect(result.text).toBe(mockResultText);
  });

  it('should not retry without errors', async () => {
    // Arrange
    const baseModel = createMockModel(mockResult);
    const fallbackModel1 = createMockModel(mockResult);
    const fallbackModel2 = createMockModel(mockResult);

    // Act
    const result = await generateText({
      model: createRetryable({
        model: baseModel,
        retries: [fallbackModel1, fallbackModel2],
      }),
      prompt: 'Hello!',
    });

    // Assert
    expect(baseModel.doGenerateCalls.length).toBe(1);
    expect(fallbackModel1.doGenerateCalls.length).toBe(0);
    expect(fallbackModel2.doGenerateCalls.length).toBe(0);
    expect(result.text).toBe(mockResultText);
  });

  it('should use fallback models for error-based attempts', async () => {
    // Arrange
    const baseModel = createMockModel(retryableError);
    const plainFallbackModel = createMockModel(mockResult);
    const functionFallbackModel = createMockModel(mockResult);

    // Act
    const result = await generateText({
      model: createRetryable({
        model: baseModel,
        retries: [
          plainFallbackModel,
          (context) => {
            if (
              isErrorAttempt(context.current) &&
              APICallError.isInstance(context.current.error)
            ) {
              return { model: functionFallbackModel, maxAttempts: 1 };
            }
            return undefined;
          },
        ],
      }),
      prompt: 'Hello!',
    });

    // Assert
    expect(baseModel.doGenerateCalls.length).toBe(1);
    expect(plainFallbackModel.doGenerateCalls.length).toBe(1); // Should be called
    expect(functionFallbackModel.doGenerateCalls.length).toBe(0);
    expect(result.text).toBe(mockResultText);
  });
});

describe('result-based retries', () => {
  it('should retry with results', async () => {
    // Arrange
    const baseModel = createMockModel(contentFilterResult);
    const fallbackModel = createMockModel(mockResult);

    // Act
    const result = await generateText({
      model: createRetryable({
        model: baseModel,
        retries: [
          (context) => {
            if (
              isResultAttempt(context.current) &&
              context.current.result.finishReason === 'content-filter'
            ) {
              return { model: fallbackModel, maxAttempts: 1 };
            }
            return undefined;
          },
        ],
      }),
      prompt: 'Hello!',
    });

    // Assert
    expect(baseModel.doGenerateCalls.length).toBe(1);
    expect(fallbackModel.doGenerateCalls.length).toBe(1);
    expect(result.text).toBe(mockResultText);
  });

  it('should ignore fallback models for result-based attempts', async () => {
    // Arrange
    const baseModel = createMockModel(contentFilterResult);
    const plainFallbackModel = createMockModel(mockResult);
    const functionFallbackModel = createMockModel(mockResult);

    // Act
    const result = await generateText({
      model: createRetryable({
        model: baseModel,
        retries: [
          plainFallbackModel, // This should be ignored for result-based retries
          (context) => {
            if (
              isResultAttempt(context.current) &&
              context.current.result.finishReason === 'content-filter'
            ) {
              return { model: functionFallbackModel, maxAttempts: 1 };
            }
            return undefined;
          },
        ],
      }),
      prompt: 'Hello!',
    });

    // Assert
    expect(baseModel.doGenerateCalls.length).toBe(1);
    expect(plainFallbackModel.doGenerateCalls.length).toBe(0); // Should not be called
    expect(functionFallbackModel.doGenerateCalls.length).toBe(1);
    expect(result.text).toBe(mockResultText);
  });
});

describe('onError', () => {
  it('should call onError handler when an error occurs', async () => {
    // Arrange
    const baseModel = createMockModel(retryableError);
    const fallbackModel = createMockModel(mockResult);
    const onErrorSpy = vi.fn<OnError>();

    // Act
    await generateText({
      model: createRetryable({
        model: baseModel,
        retries: [fallbackModel],
        onError: onErrorSpy,
      }),
      prompt: 'Hello!',
    });

    // Assert
    expect(onErrorSpy).toHaveBeenCalledTimes(1);

    const firstErrorCall = onErrorSpy.mock.calls[0]![0];
    expect(firstErrorCall.current.error).toBe(retryableError);
    expect(firstErrorCall.current.model).toBe(baseModel);
    expect(firstErrorCall.totalAttempts).toBe(1);
  });

  it('should call onError handler for each error in multiple attempts', async () => {
    // Arrange
    const baseModel = createMockModel(retryableError);
    const fallbackModel = createMockModel(nonRetryableError);
    const finalModel = createMockModel(mockResult);
    const onErrorSpy = vi.fn<OnError>();

    // Act
    await generateText({
      model: createRetryable({
        model: baseModel,
        retries: [fallbackModel, finalModel],
        onError: onErrorSpy,
      }),
      prompt: 'Hello!',
    });

    // Assert
    expect(onErrorSpy).toHaveBeenCalledTimes(2);

    // Check that onError was called for each error
    const firstErrorCall = onErrorSpy.mock.calls[0]![0];
    const secondErrorCall = onErrorSpy.mock.calls[1]![0];

    expect(firstErrorCall.current.error).toBe(retryableError);
    expect(firstErrorCall.current.model).toBe(baseModel);
    expect(firstErrorCall.totalAttempts).toBe(1);

    expect(secondErrorCall.current.error).toBe(nonRetryableError);
    expect(secondErrorCall.current.model).toBe(fallbackModel);
    expect(secondErrorCall.totalAttempts).toBe(2);
  });

  it('should not call onError handler for result-based retries', async () => {
    // Arrange
    const baseModel = createMockModel(contentFilterResult);
    const fallbackModel = createMockModel(mockResult);
    const onErrorSpy = vi.fn<OnError>();
    const onRetrySpy = vi.fn<OnRetry>();

    // Act
    await generateText({
      model: createRetryable({
        model: baseModel,
        retries: [
          (context) => {
            if (
              isResultAttempt(context.current) &&
              context.current.result.finishReason === 'content-filter'
            ) {
              return { model: fallbackModel, maxAttempts: 1 };
            }
            return undefined;
          },
        ],
        onError: onErrorSpy,
        onRetry: onRetrySpy,
      }),
      prompt: 'Hello!',
    });

    // Assert
    expect(onErrorSpy).not.toHaveBeenCalled();
    expect(onRetrySpy).toHaveBeenCalledTimes(1);
  });

  it('should call onError handler for error-based retries', async () => {
    // Arrange
    const baseModel = createMockModel(retryableError);
    const fallbackModel = createMockModel(mockResult);
    const onErrorSpy = vi.fn<OnError>();
    const onRetrySpy = vi.fn<OnRetry>();

    // Act
    await generateText({
      model: createRetryable({
        model: baseModel,
        retries: [fallbackModel],
        onError: onErrorSpy,
        onRetry: onRetrySpy,
      }),
      prompt: 'Hello!',
    });

    // Assert
    expect(onErrorSpy).toHaveBeenCalledTimes(1);
    expect(onRetrySpy).toHaveBeenCalledTimes(1);

    // Verify onError is called before onRetry by checking call order
    const errorCallTime = onErrorSpy.mock.invocationCallOrder[0] ?? 0;
    const retryCallTime = onRetrySpy.mock.invocationCallOrder[0] ?? 0;
    expect(errorCallTime).toBeLessThan(retryCallTime);
  });
});

describe('onRetry', () => {
  it('should call onRetry handler for error-based retries', async () => {
    // Arrange
    const baseModel = createMockModel(retryableError);
    const fallbackModel = createMockModel(mockResult);
    const onRetrySpy = vi.fn<OnRetry>();

    // Act
    await generateText({
      model: createRetryable({
        model: baseModel,
        retries: [fallbackModel],
        onRetry: onRetrySpy,
      }),
      prompt: 'Hello!',
    });

    // Assert
    expect(onRetrySpy).toHaveBeenCalledTimes(1);

    const firstRetryCall = onRetrySpy.mock.calls[0]![0];
    expect(isErrorAttempt(firstRetryCall.current)).toBe(true);
    if (isErrorAttempt(firstRetryCall.current)) {
      expect(firstRetryCall.current.error).toBe(retryableError);
    }
    expect(firstRetryCall.current.model).toBe(fallbackModel);
    expect(firstRetryCall.totalAttempts).toBe(1);
  });

  it('should call onRetry handler for result-based retries', async () => {
    // Arrange
    const baseModel = createMockModel(contentFilterResult);
    const fallbackModel = createMockModel(mockResult);
    const onRetrySpy = vi.fn<OnRetry>();

    // Act
    await generateText({
      model: createRetryable({
        model: baseModel,
        retries: [
          (context) => {
            if (
              isResultAttempt(context.current) &&
              context.current.result.finishReason === 'content-filter'
            ) {
              return { model: fallbackModel, maxAttempts: 1 };
            }
            return undefined;
          },
        ],
        onRetry: onRetrySpy,
      }),
      prompt: 'Hello!',
    });

    // Assert
    expect(onRetrySpy).toHaveBeenCalledTimes(1);

    const retryCall = onRetrySpy.mock.calls[0]![0];
    expect(isResultAttempt(retryCall.current)).toBe(true);
    if (isResultAttempt(retryCall.current)) {
      expect(retryCall.current.result.finishReason).toBe('content-filter');
    }
    expect(retryCall.current.model).toBe(fallbackModel);
    expect(retryCall.totalAttempts).toBe(1);
  });

  it('should call onRetry handler for each retry attempt', async () => {
    // Arrange
    const baseModel = createMockModel(retryableError);
    const fallbackModel1 = createMockModel(nonRetryableError);
    const fallbackModel2 = createMockModel(mockResult);
    const onRetrySpy = vi.fn<OnRetry>();

    // Act
    await generateText({
      model: createRetryable({
        model: baseModel,
        retries: [fallbackModel1, fallbackModel2],
        onRetry: onRetrySpy,
      }),
      prompt: 'Hello!',
    });

    // Assert
    expect(onRetrySpy).toHaveBeenCalledTimes(2);

    // Check that onRetry was called for each retry
    const firstRetryCall = onRetrySpy.mock.calls[0]![0];
    const secondRetryCall = onRetrySpy.mock.calls[1]![0];

    expect(isErrorAttempt(firstRetryCall.current)).toBe(true);
    if (isErrorAttempt(firstRetryCall.current)) {
      expect(firstRetryCall.current.error).toBe(retryableError);
    }
    expect(firstRetryCall.current.model).toBe(fallbackModel1);
    expect(firstRetryCall.totalAttempts).toBe(1);

    expect(isErrorAttempt(secondRetryCall.current)).toBe(true);
    if (isErrorAttempt(secondRetryCall.current)) {
      expect(secondRetryCall.current.error).toBe(nonRetryableError);
    }
    expect(secondRetryCall.current.model).toBe(fallbackModel2);
    expect(secondRetryCall.totalAttempts).toBe(2);
  });

  it('should not call onRetry on first attempt', async () => {
    // Arrange
    const baseModel = createMockModel(mockResult);
    const onRetrySpy = vi.fn<OnRetry>();

    // Act
    await generateText({
      model: createRetryable({
        model: baseModel,
        retries: [],
        onRetry: onRetrySpy,
      }),
      prompt: 'Hello!',
    });

    // Assert
    expect(onRetrySpy).not.toHaveBeenCalled();
  });
});

describe('maxAttempts', () => {
  it('should try each model only once by default', async () => {
    // Arrange
    const baseModel = createMockModel(retryableError);
    const fallbackModel1 = createMockModel(retryableError);
    const fallbackModel2 = createMockModel(retryableError);
    const finalModel = createMockModel(mockResult);

    // Act
    await generateText({
      model: createRetryable({
        model: baseModel,
        retries: [
          fallbackModel1,
          () => ({ model: fallbackModel2 }),
          async () => ({ model: finalModel }),
        ],
      }),
      prompt: 'Hello!',
    });

    // Assert
    expect(baseModel.doGenerateCalls.length).toBe(1);
    expect(fallbackModel1.doGenerateCalls.length).toBe(1);
    expect(fallbackModel2.doGenerateCalls.length).toBe(1);
    expect(finalModel.doGenerateCalls.length).toBe(1);
  });

  it('should try models multiple times if maxAttempts is set', async () => {
    // Arrange
    const baseModel = createMockModel(retryableError);
    const fallbackModel1 = createMockModel(retryableError);
    const fallbackModel2 = createMockModel(retryableError);
    const finalModel = createMockModel(mockResult);

    // Act
    await generateText({
      model: createRetryable({
        model: baseModel,
        retries: [
          // Retryable with different maxAttempts
          () => ({ model: fallbackModel1, maxAttempts: 2 }),
          async () => ({ model: fallbackModel2, maxAttempts: 3 }),
          finalModel,
        ],
      }),
      prompt: 'Hello!',
    });

    // Assert
    expect(baseModel.doGenerateCalls.length).toBe(1);
    expect(fallbackModel1.doGenerateCalls.length).toBe(2);
    expect(fallbackModel2.doGenerateCalls.length).toBe(3);
    expect(finalModel.doGenerateCalls.length).toBe(1);
  });
});

describe('maxRetries', () => {
  it('should ignore maxRetries setting when retryable matches', async () => {
    // Arrange
    const baseModel = createMockModel(retryableError);
    const fallbackModel = createMockModel(nonRetryableError);

    // Act & Assert
    try {
      await generateText({
        model: createRetryable({
          model: baseModel,
          retries: [fallbackModel],
        }),
        prompt: 'Hello!',
        maxRetries: 1, // Should be ignored since RetryError is thrown
      });
      expect.unreachable('Should throw RetryError');
    } catch (error) {
      expect(error).toBeInstanceOf(RetryError);
      expect(baseModel.doGenerateCalls.length).toBe(1);
      expect(fallbackModel.doGenerateCalls.length).toBe(1);
    }
  });

  it('should respect maxRetries setting when no retryable matches', async () => {
    // Arrange
    const baseModel = createMockModel(retryableError);

    // Act & Assert
    try {
      await generateText({
        model: createRetryable({
          model: baseModel,
          retries: [],
        }),
        prompt: 'Hello!',
        maxRetries: 1, // Should be ignored since RetryError is thrown
      });
      expect.unreachable('Should throw RetryError');
    } catch (error) {
      expect(error).toBeInstanceOf(RetryError);
      expect(baseModel.doGenerateCalls.length).toBe(2); // 1 initial + 1 retry
    }
  });
});

describe('RetryError', () => {
  it('should throw RetryError when all retry attempts are exhausted', async () => {
    // Arrange
    const baseModel = createMockModel(retryableError);
    const fallbackModel1 = createMockModel(nonRetryableError);
    const fallbackModel2 = createMockModel(retryableError);

    // Act & Assert
    try {
      await generateText({
        model: createRetryable({
          model: baseModel,
          retries: [fallbackModel1, fallbackModel2],
        }),
        prompt: 'Hello!',
      });
      expect.unreachable(
        'Should throw RetryError when all attempts are exhausted',
      );
    } catch (error) {
      expect(error).toBeInstanceOf(RetryError);

      const retryError = error as RetryError;
      expect(retryError.reason).toBe('maxRetriesExceeded');
      expect(retryError.errors).toHaveLength(3);
      expect(retryError.errors[0]).toBe(retryableError);
      expect(retryError.errors[1]).toBe(nonRetryableError);
      expect(retryError.errors[2]).toBe(retryableError);
    }
  });

  it('should throw original error directly on first attempt with no retryables', async () => {
    // Arrange
    const baseModel = createMockModel(retryableError);

    // Act & Assert
    try {
      await generateText({
        model: createRetryable({
          model: baseModel,
          retries: [], // No retry models
        }),
        prompt: 'Hello!',
        maxRetries: 0, // No automatic retries
      });
      expect.unreachable(
        'Should throw original error on first attempt with no retries',
      );
    } catch (error) {
      expect(error).not.toBeInstanceOf(RetryError);
      expect(error).toBe(retryableError);
    }
  });

  it('should throw original error directly when retryable returns undefined', async () => {
    // Arrange
    const baseModel = createMockModel(nonRetryableError);

    // Act & Assert
    try {
      await generateText({
        model: createRetryable({
          model: baseModel,
          retries: [() => undefined],
        }),
        prompt: 'Hello!',
        maxRetries: 0, // No automatic retries
      });
      expect.unreachable(
        'Should throw original error when retry models return undefined',
      );
    } catch (error) {
      expect(error).not.toBeInstanceOf(RetryError);
      expect(error).toBe(nonRetryableError);
    }
  });
});
