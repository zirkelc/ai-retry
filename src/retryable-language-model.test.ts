import {
  convertArrayToReadableStream,
  convertAsyncIterableToArray,
} from '@ai-sdk/provider-utils/test';
import { APICallError, generateText, RetryError, streamText } from 'ai';
import { describe, expect, it, vi } from 'vitest';
import { createRetryable } from './create-retryable-model.js';
import {
  chunksToText,
  errorFromChunks,
  MockLanguageModel,
  mockStreamOptions,
} from './test-utils.js';
import type {
  LanguageModel,
  LanguageModelCallOptions,
  LanguageModelGenerate,
  LanguageModelStreamPart,
  Retryable,
  RetryableModelOptions,
  RetryContext,
} from './types.js';
import { isErrorAttempt, isResultAttempt } from './utils.js';

type OnError = Required<RetryableModelOptions<LanguageModel>>['onError'];
type OnRetry = Required<RetryableModelOptions<LanguageModel>>['onRetry'];

const prompt = 'Hello!';

const mockResultText = 'Hello, world!';

const mockResult: LanguageModelGenerate = {
  finishReason: 'stop',
  usage: {
    inputTokens: { total: 10, noCache: 0, cacheRead: 0, cacheWrite: 0 },
    outputTokens: { total: 20, text: 0, reasoning: 0 },
  },
  content: [{ type: 'text', text: mockResultText }],
  warnings: [],
};

const contentFilterResult: LanguageModelGenerate = {
  finishReason: 'content-filter',
  usage: {
    inputTokens: { total: 10, noCache: 0, cacheRead: 0, cacheWrite: 0 },
    outputTokens: { total: 20, text: 0, reasoning: 0 },
  },
  content: [],
  warnings: [],
};

const testUsage = {
  inputTokens: { total: 3, noCache: 0, cacheRead: 0, cacheWrite: 0 },
  outputTokens: { total: 10, text: 0, reasoning: 0 },
};

const mockStreamChunks: LanguageModelStreamPart[] = [
  {
    type: 'stream-start',
    warnings: [],
  },
  {
    type: 'response-metadata',
    id: 'id-0',
    modelId: 'mock-model-id',
    timestamp: new Date(0),
  },
  { type: 'text-start', id: '1' },
  { type: 'text-delta', id: '1', delta: 'Hello' },
  { type: 'text-delta', id: '1', delta: ', ' },
  { type: 'text-delta', id: '1', delta: `world!` },
  { type: 'text-end', id: '1' },
  {
    type: 'finish',
    finishReason: 'stop',
    usage: testUsage,
    providerMetadata: {
      testProvider: { testKey: 'testValue' },
    },
  },
];

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

describe('generateText', () => {
  it('should generate text successfully when no errors occur', async () => {
    // Arrange
    const baseModel = new MockLanguageModel({
      doGenerate: {
        finishReason: 'stop',
        usage: {
          inputTokens: { total: 10, noCache: 0, cacheRead: 0, cacheWrite: 0 },
          outputTokens: { total: 20, text: 0, reasoning: 0 },
        },
        content: [{ type: 'text', text: 'Hello, world!' }],
        warnings: [],
      },
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
    expect(baseModel.doGenerate).toHaveBeenCalledTimes(1);
    expect(result.text).toBe('Hello, world!');
  });

  describe('retries', () => {
    describe('error-based retries', () => {
      it('should retry with errors', async () => {
        // Arrange
        const baseModel = new MockLanguageModel({ doGenerate: retryableError });
        const fallbackModel = new MockLanguageModel({ doGenerate: mockResult });

        const fallbackRetryable: Retryable<LanguageModel> = (context) => {
          if (
            isErrorAttempt(context.current) &&
            APICallError.isInstance(context.current.error)
          ) {
            return { model: fallbackModel, maxAttempts: 1 };
          }
          return undefined;
        };

        // Act
        const result = await generateText({
          model: createRetryable({
            model: baseModel,
            retries: [fallbackRetryable],
          }),
          prompt: 'Hello!',
          //
        });

        // Assert
        expect(baseModel.doGenerate).toHaveBeenCalledTimes(1);
        expect(fallbackModel.doGenerate).toHaveBeenCalledTimes(1);
        expect(result.text).toBe(mockResultText);
      });

      it('should not retry without errors', async () => {
        // Arrange
        const baseModel = new MockLanguageModel({ doGenerate: mockResult });
        const fallbackModel1 = new MockLanguageModel({
          doGenerate: mockResult,
        });
        const fallbackModel2 = new MockLanguageModel({
          doGenerate: mockResult,
        });

        // Act
        const result = await generateText({
          model: createRetryable({
            model: baseModel,
            retries: [fallbackModel1, fallbackModel2],
          }),
          prompt: 'Hello!',
        });

        // Assert
        expect(baseModel.doGenerate).toHaveBeenCalledTimes(1);
        expect(fallbackModel1.doGenerate).toHaveBeenCalledTimes(0);
        expect(fallbackModel2.doGenerate).toHaveBeenCalledTimes(0);
        expect(result.text).toBe(mockResultText);
      });

      it('should use plain language models for error-based attempts', async () => {
        // Arrange
        const baseModel = new MockLanguageModel({ doGenerate: retryableError });
        const fallbackModel1 = new MockLanguageModel({
          doGenerate: mockResult,
        });
        const fallbackModel2 = new MockLanguageModel({
          doGenerate: mockResult,
        });

        const fallbackRetryable: Retryable<LanguageModel> = (context) => {
          if (
            isErrorAttempt(context.current) &&
            APICallError.isInstance(context.current.error)
          ) {
            return { model: fallbackModel2, maxAttempts: 1 };
          }
          return undefined;
        };

        // Act
        const result = await generateText({
          model: createRetryable({
            model: baseModel,
            retries: [fallbackModel1, fallbackRetryable],
          }),
          prompt: 'Hello!',
        });

        // Assert
        expect(baseModel.doGenerate).toHaveBeenCalledTimes(1);
        expect(fallbackModel1.doGenerate).toHaveBeenCalledTimes(1); // Should be called
        expect(fallbackModel2.doGenerate).toHaveBeenCalledTimes(0);
        expect(result.text).toBe(mockResultText);
      });

      it('should use static retry for error-based attempts', async () => {
        // Arrange
        const baseModel = new MockLanguageModel({
          doGenerate: retryableError,
        });
        const fallbackModel1 = new MockLanguageModel({
          doGenerate: mockResult,
        });
        const fallbackModel2 = new MockLanguageModel({
          doGenerate: mockResult,
        });

        const fallbackRetryable: Retryable<LanguageModel> = (context) => {
          if (
            isErrorAttempt(context.current) &&
            APICallError.isInstance(context.current.error)
          ) {
            return { model: fallbackModel2, maxAttempts: 1 };
          }
          return undefined;
        };

        // Act
        const result = await generateText({
          model: createRetryable({
            model: baseModel,
            retries: [{ model: fallbackModel1 }, fallbackRetryable],
          }),
          prompt: 'Hello!',
        });

        // Assert
        expect(baseModel.doGenerate).toHaveBeenCalledTimes(1);
        expect(fallbackModel1.doGenerate).toHaveBeenCalledTimes(1); // Should be called
        expect(fallbackModel2.doGenerate).toHaveBeenCalledTimes(0);
        expect(result.text).toBe(mockResultText);
      });
    });

    describe('result-based retries', () => {
      it('should retry with results', async () => {
        // Arrange
        const baseModel = new MockLanguageModel({
          doGenerate: contentFilterResult,
        });
        const fallbackModel = new MockLanguageModel({ doGenerate: mockResult });
        const fallbackRetryable: Retryable<LanguageModel> = (context) => {
          if (
            isResultAttempt(context.current) &&
            context.current.result.finishReason === 'content-filter'
          ) {
            return { model: fallbackModel, maxAttempts: 1 };
          }
          return undefined;
        };

        // Act
        const result = await generateText({
          model: createRetryable({
            model: baseModel,
            retries: [fallbackRetryable],
          }),
          prompt: 'Hello!',
        });

        // Assert
        expect(baseModel.doGenerate).toHaveBeenCalledTimes(1);
        expect(fallbackModel.doGenerate).toHaveBeenCalledTimes(1);
        expect(result.text).toBe(mockResultText);
      });

      it('should ignore plain language models for result-based attempts', async () => {
        // Arrange
        const baseModel = new MockLanguageModel({
          doGenerate: contentFilterResult,
        });
        const fallbackModel1 = new MockLanguageModel({
          doGenerate: mockResult,
        });
        const fallbackModel2 = new MockLanguageModel({
          doGenerate: mockResult,
        });

        const fallbackRetryable: Retryable<LanguageModel> = (context) => {
          if (
            isResultAttempt(context.current) &&
            context.current.result.finishReason === 'content-filter'
          ) {
            return { model: fallbackModel2, maxAttempts: 1 };
          }
          return undefined;
        };

        // Act
        const result = await generateText({
          model: createRetryable({
            model: baseModel,
            retries: [
              fallbackModel1, // Language model should be skipped
              fallbackRetryable,
            ],
          }),
          prompt: 'Hello!',
        });

        // Assert
        expect(baseModel.doGenerate).toHaveBeenCalledTimes(1);
        expect(fallbackModel1.doGenerate).toHaveBeenCalledTimes(0); // Should not be called
        expect(fallbackModel2.doGenerate).toHaveBeenCalledTimes(1);
        expect(result.text).toBe(mockResultText);
      });

      it('should ignore static retries for result-based attempts', async () => {
        // Arrange
        const baseModel = new MockLanguageModel({
          doGenerate: contentFilterResult,
        });
        const fallbackModel1 = new MockLanguageModel({
          doGenerate: mockResult,
        });
        const fallbackModel2 = new MockLanguageModel({
          doGenerate: mockResult,
        });

        const fallbackRetryable: Retryable<LanguageModel> = (context) => {
          if (
            isResultAttempt(context.current) &&
            context.current.result.finishReason === 'content-filter'
          ) {
            return { model: fallbackModel2, maxAttempts: 1 };
          }
          return undefined;
        };

        // Act
        const result = await generateText({
          model: createRetryable({
            model: baseModel,
            retries: [
              { model: fallbackModel1 }, // Static retry should be skipped
              fallbackRetryable,
            ],
          }),
          prompt: 'Hello!',
        });

        // Assert
        expect(baseModel.doGenerate).toHaveBeenCalledTimes(1);
        expect(fallbackModel1.doGenerate).toHaveBeenCalledTimes(0); // Should not be called
        expect(fallbackModel2.doGenerate).toHaveBeenCalledTimes(1);
        expect(result.text).toBe(mockResultText);
      });
    });
  });

  describe('disabled', () => {
    it('should not retry when disabled is true (boolean)', async () => {
      // Arrange
      const baseModel = new MockLanguageModel({ doGenerate: retryableError });
      const fallbackModel = new MockLanguageModel({ doGenerate: mockResult });

      const fallbackRetryable: Retryable<LanguageModel> = () => {
        return { model: fallbackModel, maxAttempts: 1 };
      };

      const retryableModel = createRetryable({
        model: baseModel,
        retries: [fallbackRetryable],
        disabled: true,
      });

      // Act & Assert
      await expect(
        generateText({
          model: retryableModel,
          prompt: 'Hello!',
          maxRetries: 0, // Disable AI SDK's own retry mechanism
        }),
      ).rejects.toThrow('Rate limit exceeded');

      expect(baseModel.doGenerate).toHaveBeenCalledTimes(1);
      expect(fallbackModel.doGenerate).toHaveBeenCalledTimes(0);
    });

    it('should retry when disabled is false (boolean)', async () => {
      // Arrange
      const baseModel = new MockLanguageModel({ doGenerate: retryableError });
      const fallbackModel = new MockLanguageModel({ doGenerate: mockResult });

      const fallbackRetryable: Retryable<LanguageModel> = () => {
        return { model: fallbackModel, maxAttempts: 1 };
      };

      const retryableModel = createRetryable({
        model: baseModel,
        retries: [fallbackRetryable],
        disabled: false,
      });

      // Act
      const result = await generateText({
        model: retryableModel,
        prompt: 'Hello!',
      });

      // Assert
      expect(baseModel.doGenerate).toHaveBeenCalledTimes(1);
      expect(fallbackModel.doGenerate).toHaveBeenCalledTimes(1);
      expect(result.text).toBe('Hello, world!');
    });

    it('should not retry when disabled is a function returning true', async () => {
      // Arrange
      const baseModel = new MockLanguageModel({ doGenerate: retryableError });
      const fallbackModel = new MockLanguageModel({ doGenerate: mockResult });

      const fallbackRetryable: Retryable<LanguageModel> = () => {
        return { model: fallbackModel, maxAttempts: 1 };
      };

      const disabledFn = vi.fn(() => true);

      const retryableModel = createRetryable({
        model: baseModel,
        retries: [fallbackRetryable],
        disabled: disabledFn,
      });

      // Act & Assert
      await expect(
        generateText({
          model: retryableModel,
          prompt: 'Hello!',
          maxRetries: 0, // Disable AI SDK's own retry mechanism
        }),
      ).rejects.toThrow('Rate limit exceeded');

      expect(disabledFn).toHaveBeenCalled();
      expect(baseModel.doGenerate).toHaveBeenCalledTimes(1);
      expect(fallbackModel.doGenerate).toHaveBeenCalledTimes(0);
    });

    it('should retry when disabled is a function returning false', async () => {
      // Arrange
      const baseModel = new MockLanguageModel({ doGenerate: retryableError });
      const fallbackModel = new MockLanguageModel({ doGenerate: mockResult });

      const fallbackRetryable: Retryable<LanguageModel> = () => {
        return { model: fallbackModel, maxAttempts: 1 };
      };

      const disabledFn = vi.fn(() => false);

      const retryableModel = createRetryable({
        model: baseModel,
        retries: [fallbackRetryable],
        disabled: disabledFn,
      });

      // Act
      const result = await generateText({
        model: retryableModel,
        prompt: 'Hello!',
      });

      // Assert
      expect(disabledFn).toHaveBeenCalled();
      expect(baseModel.doGenerate).toHaveBeenCalledTimes(1);
      expect(fallbackModel.doGenerate).toHaveBeenCalledTimes(1);
      expect(result.text).toBe('Hello, world!');
    });

    it('should work normally when disabled is undefined (default)', async () => {
      // Arrange
      const baseModel = new MockLanguageModel({ doGenerate: retryableError });
      const fallbackModel = new MockLanguageModel({ doGenerate: mockResult });

      const fallbackRetryable: Retryable<LanguageModel> = () => {
        return { model: fallbackModel, maxAttempts: 1 };
      };

      const retryableModel = createRetryable({
        model: baseModel,
        retries: [fallbackRetryable],
        // disabled is undefined by default
      });

      // Act
      const result = await generateText({
        model: retryableModel,
        prompt: 'Hello!',
      });

      // Assert
      expect(baseModel.doGenerate).toHaveBeenCalledTimes(1);
      expect(fallbackModel.doGenerate).toHaveBeenCalledTimes(1);
      expect(result.text).toBe('Hello, world!');
    });
  });

  describe('onError', () => {
    it('should call onError handler when an error occurs', async () => {
      // Arrange
      const baseModel = new MockLanguageModel({ doGenerate: retryableError });
      const fallbackModel = new MockLanguageModel({ doGenerate: mockResult });
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
      expect(firstErrorCall.attempts.length).toBe(1);
      // expect(firstErrorCall.totalAttempts).toBe(1);
    });

    it('should call onError handler for each error in multiple attempts', async () => {
      // Arrange
      const baseModel = new MockLanguageModel({ doGenerate: retryableError });
      const fallbackModel = new MockLanguageModel({
        doGenerate: nonRetryableError,
      });
      const finalModel = new MockLanguageModel({ doGenerate: mockResult });
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
      expect(firstErrorCall.attempts.length).toBe(1);
      // expect(firstErrorCall.totalAttempts).toBe(1);

      expect(secondErrorCall.current.error).toBe(nonRetryableError);
      expect(secondErrorCall.current.model).toBe(fallbackModel);
      expect(secondErrorCall.attempts.length).toBe(2);
      // expect(secondErrorCall.totalAttempts).toBe(2);
    });

    it('should NOT call onError handler for result-based retries', async () => {
      // Arrange
      const baseModel = new MockLanguageModel({
        doGenerate: contentFilterResult,
      });
      const fallbackModel = new MockLanguageModel({ doGenerate: mockResult });
      const onErrorSpy = vi.fn<OnError>();
      const onRetrySpy = vi.fn<OnRetry>();

      // Act
      await generateText({
        model: createRetryable({
          model: baseModel,
          retries: [
            (context: RetryContext<LanguageModel>) => {
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

    it('should call onError handler before onRetry handler', async () => {
      // Arrange
      const baseModel = new MockLanguageModel({ doGenerate: retryableError });
      const fallbackModel = new MockLanguageModel({ doGenerate: mockResult });
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

      // Verify the context passed to each handler
      const firstErrorCall = onErrorSpy.mock.calls[0]![0];
      const firstRetryCall = onRetrySpy.mock.calls[0]![0];
      expect(firstErrorCall.current.model).toBe(baseModel);
      expect(firstErrorCall.attempts.length).toBe(1);
      expect(firstRetryCall.current.model).toBe(fallbackModel);
      expect(firstRetryCall.attempts.length).toBe(1);
    });
  });

  describe('onRetry', () => {
    it('should call onRetry handler for error-based retries', async () => {
      // Arrange
      const baseModel = new MockLanguageModel({ doGenerate: retryableError });
      const fallbackModel = new MockLanguageModel({ doGenerate: mockResult });
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
      expect(firstRetryCall.attempts.length).toBe(1);
      // expect(firstRetryCall.totalAttempts).toBe(1);
    });

    it('should call onRetry handler for result-based retries', async () => {
      // Arrange
      const baseModel = new MockLanguageModel({
        doGenerate: contentFilterResult,
      });
      const fallbackModel = new MockLanguageModel({ doGenerate: mockResult });
      const onRetrySpy = vi.fn<OnRetry>();

      // Act
      await generateText({
        model: createRetryable({
          model: baseModel,
          retries: [
            (context: RetryContext<LanguageModel>) => {
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
      expect(retryCall.attempts.length).toBe(1);
      // expect(retryCall.totalAttempts).toBe(1);
    });

    it('should call onRetry handler for each retry attempt', async () => {
      // Arrange
      const baseModel = new MockLanguageModel({ doGenerate: retryableError });
      const fallbackModel1 = new MockLanguageModel({
        doGenerate: nonRetryableError,
      });
      const fallbackModel2 = new MockLanguageModel({ doGenerate: mockResult });
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
      expect(firstRetryCall.attempts.length).toBe(1);
      // expect(firstRetryCall.totalAttempts).toBe(1);

      expect(isErrorAttempt(secondRetryCall.current)).toBe(true);
      if (isErrorAttempt(secondRetryCall.current)) {
        expect(secondRetryCall.current.error).toBe(nonRetryableError);
      }
      expect(secondRetryCall.current.model).toBe(fallbackModel2);
      expect(secondRetryCall.attempts.length).toBe(2);
      // expect(secondRetryCall.totalAttempts).toBe(2);
    });

    it('should NOT call onRetry on first attempt', async () => {
      // Arrange
      const baseModel = new MockLanguageModel({ doGenerate: mockResult });
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

  describe('attempt options', () => {
    it('should include call options in error attempts', async () => {
      // Arrange
      const baseModel = new MockLanguageModel({ doGenerate: retryableError });
      const fallbackModel = new MockLanguageModel({ doGenerate: mockResult });
      const onErrorSpy = vi.fn<OnError>();

      // Act
      await generateText({
        model: createRetryable({
          model: baseModel,
          retries: [fallbackModel],
          onError: onErrorSpy,
        }),
        prompt: 'Hello!',
        temperature: 0.7,
        maxOutputTokens: 1000,
      });

      // Assert
      expect(onErrorSpy).toHaveBeenCalledTimes(1);

      const errorContext = onErrorSpy.mock.calls[0]![0];
      expect(errorContext.current.options).toBeDefined();
      expect(errorContext.current.options.temperature).toBe(0.7);
      expect(errorContext.current.options.maxOutputTokens).toBe(1000);
    });

    it('should include call options in result attempts', async () => {
      // Arrange
      const baseModel = new MockLanguageModel({
        doGenerate: contentFilterResult,
      });
      const fallbackModel = new MockLanguageModel({ doGenerate: mockResult });
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
        temperature: 0.8,
        seed: 42,
      });

      // Assert
      expect(onRetrySpy).toHaveBeenCalledTimes(1);

      const retryContext = onRetrySpy.mock.calls[0]![0];
      expect(isResultAttempt(retryContext.current)).toBe(true);
      if (isResultAttempt(retryContext.current)) {
        expect(retryContext.current.options).toBeDefined();
        expect(retryContext.current.options.temperature).toBe(0.8);
        expect(retryContext.current.options.seed).toBe(42);
      }
    });

    it('should reflect overridden options in retry attempts', async () => {
      // Arrange
      const baseModel = new MockLanguageModel({ doGenerate: retryableError });
      const fallbackModel = new MockLanguageModel({
        doGenerate: nonRetryableError,
      });
      const finalModel = new MockLanguageModel({ doGenerate: mockResult });
      const onErrorSpy = vi.fn<OnError>();

      // Act
      await generateText({
        model: createRetryable({
          model: baseModel,
          retries: [
            { model: fallbackModel, options: { temperature: 0.5 } },
            finalModel,
          ],
          onError: onErrorSpy,
        }),
        prompt: 'Hello!',
        temperature: 1.0,
      });

      // Assert
      expect(onErrorSpy).toHaveBeenCalledTimes(2);

      // First attempt should have original temperature
      const firstErrorContext = onErrorSpy.mock.calls[0]![0];
      expect(firstErrorContext.current.options.temperature).toBe(1.0);

      // Second attempt should have overridden temperature
      const secondErrorContext = onErrorSpy.mock.calls[1]![0];
      expect(secondErrorContext.current.options.temperature).toBe(0.5);
    });

    it('should include prompt in options', async () => {
      // Arrange
      const baseModel = new MockLanguageModel({ doGenerate: retryableError });
      const fallbackModel = new MockLanguageModel({ doGenerate: mockResult });
      const onErrorSpy = vi.fn<OnError>();

      // Act
      await generateText({
        model: createRetryable({
          model: baseModel,
          retries: [fallbackModel],
          onError: onErrorSpy,
        }),
        prompt: 'Test prompt',
      });

      // Assert
      expect(onErrorSpy).toHaveBeenCalledTimes(1);

      const errorContext = onErrorSpy.mock.calls[0]![0];
      expect(errorContext.current.options.prompt).toBeDefined();
      expect(errorContext.current.options.prompt).toEqual(
        expect.arrayContaining([
          expect.objectContaining({
            role: 'user',
            content: expect.arrayContaining([
              expect.objectContaining({ type: 'text', text: 'Test prompt' }),
            ]),
          }),
        ]),
      );
    });
  });

  describe('RetryableOptions', () => {
    describe('maxAttempts', () => {
      it('should try each model only once by default', async () => {
        // Arrange
        const baseModel = new MockLanguageModel({ doGenerate: retryableError });
        const fallbackModel1 = new MockLanguageModel({
          doGenerate: retryableError,
        });
        const fallbackModel2 = new MockLanguageModel({
          doGenerate: retryableError,
        });
        const finalModel = new MockLanguageModel({ doGenerate: mockResult });

        // Act
        await generateText({
          model: createRetryable({
            model: baseModel,
            retries: [
              fallbackModel1,
              { model: fallbackModel2 },
              async () => ({ model: finalModel }),
            ],
          }),
          prompt: 'Hello!',
        });

        // Assert
        expect(baseModel.doGenerate).toHaveBeenCalledTimes(1);
        expect(fallbackModel1.doGenerate).toHaveBeenCalledTimes(1);
        expect(fallbackModel2.doGenerate).toHaveBeenCalledTimes(1);
        expect(finalModel.doGenerate).toHaveBeenCalledTimes(1);
      });

      it('should try models multiple times if maxAttempts is set', async () => {
        // Arrange
        const baseModel = new MockLanguageModel({ doGenerate: retryableError });
        const fallbackModel1 = new MockLanguageModel({
          doGenerate: retryableError,
        });
        const fallbackModel2 = new MockLanguageModel({
          doGenerate: retryableError,
        });
        const finalModel = new MockLanguageModel({ doGenerate: mockResult });

        // Act
        await generateText({
          model: createRetryable({
            model: baseModel,
            retries: [
              // Retryable<LanguageModel>  with different maxAttempts
              { model: fallbackModel1, maxAttempts: 2 },
              async () => ({ model: fallbackModel2, maxAttempts: 3 }),
              finalModel,
            ],
          }),
          prompt: 'Hello!',
        });

        // Assert
        expect(baseModel.doGenerate).toHaveBeenCalledTimes(1);
        expect(fallbackModel1.doGenerate).toHaveBeenCalledTimes(2);
        expect(fallbackModel2.doGenerate).toHaveBeenCalledTimes(3);
        expect(finalModel.doGenerate).toHaveBeenCalledTimes(1);
      });
    });

    describe('maxRetries', () => {
      it('should ignore maxRetries setting when retryable matches', async () => {
        // Arrange
        const baseModel = new MockLanguageModel({ doGenerate: retryableError });
        const fallbackModel = new MockLanguageModel({
          doGenerate: nonRetryableError,
        });

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
          expect(baseModel.doGenerate).toHaveBeenCalledTimes(1);
          expect(fallbackModel.doGenerate).toHaveBeenCalledTimes(1);
        }
      });

      it('should respect maxRetries setting when no retryable matches', async () => {
        // Arrange
        const baseModel = new MockLanguageModel({ doGenerate: retryableError });

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
          expect(baseModel.doGenerate).toHaveBeenCalledTimes(2); // 1 initial + 1 retry
        }
      });
    });

    describe('delay', () => {
      it('should apply delay before retrying', async () => {
        // Arrange
        vi.useFakeTimers();
        const baseModel = new MockLanguageModel({ doGenerate: retryableError });
        const fallbackModel = new MockLanguageModel({ doGenerate: mockResult });
        const delayMs = 100;

        // Act
        const promise = generateText({
          model: createRetryable({
            model: baseModel,
            retries: [{ model: fallbackModel, delay: delayMs }],
          }),
          prompt: 'Hello!',
        });

        await vi.runAllTimersAsync();
        await promise;

        // Assert
        expect(baseModel.doGenerate).toHaveBeenCalledTimes(1);
        expect(fallbackModel.doGenerate).toHaveBeenCalledTimes(1);

        vi.useRealTimers();
      });

      it('should apply different delays for multiple retries', async () => {
        // Arrange
        vi.useFakeTimers();
        const baseModel = new MockLanguageModel({ doGenerate: retryableError });
        const fallbackModel1 = new MockLanguageModel({
          doGenerate: retryableError,
        });
        const fallbackModel2 = new MockLanguageModel({
          doGenerate: mockResult,
        });
        const delay1 = 50;
        const delay2 = 50;

        // Act
        const promise = generateText({
          model: createRetryable({
            model: baseModel,
            retries: [
              { model: fallbackModel1, delay: delay1 },
              () => ({ model: fallbackModel2, delay: delay2 }),
            ],
          }),
          prompt: 'Hello!',
        });

        await vi.runAllTimersAsync();
        await promise;

        // Assert
        expect(baseModel.doGenerate).toHaveBeenCalledTimes(1);
        expect(fallbackModel1.doGenerate).toHaveBeenCalledTimes(1);
        expect(fallbackModel2.doGenerate).toHaveBeenCalledTimes(1);

        vi.useRealTimers();
      });

      it('should not delay when delay is not specified', async () => {
        // Arrange
        const baseModel = new MockLanguageModel({ doGenerate: retryableError });
        const fallbackModel = new MockLanguageModel({ doGenerate: mockResult });

        // Act
        await generateText({
          model: createRetryable({
            model: baseModel,
            retries: [{ model: fallbackModel }],
          }),
          prompt: 'Hello!',
        });

        // Assert
        expect(baseModel.doGenerate).toHaveBeenCalledTimes(1);
        expect(fallbackModel.doGenerate).toHaveBeenCalledTimes(1);
      });
    });

    describe('providerOptions', () => {
      it('should override base model providerOptions with retry model providerOptions', async () => {
        // Arrange
        const baseModel = new MockLanguageModel({ doGenerate: retryableError });
        const fallbackModel = new MockLanguageModel({ doGenerate: mockResult });
        const originalProviderOptions = { openai: { user: 'original-user' } };
        const retryProviderOptions = { openai: { user: 'retry-user' } };

        // Act
        await generateText({
          model: createRetryable({
            model: baseModel,
            retries: [
              {
                model: fallbackModel,
                providerOptions: retryProviderOptions,
              },
            ],
          }),
          prompt: 'Hello!',
          providerOptions: originalProviderOptions,
        });

        // Assert
        expect(baseModel.doGenerate).toHaveBeenCalledTimes(1);
        expect(fallbackModel.doGenerate).toHaveBeenCalledTimes(1);
        expect(baseModel.doGenerate).toHaveBeenCalledWith(
          expect.objectContaining({
            providerOptions: originalProviderOptions,
          }),
        );
        expect(fallbackModel.doGenerate).toHaveBeenCalledWith(
          expect.objectContaining({
            providerOptions: retryProviderOptions,
          }),
        );
      });
    });

    describe('timeout', () => {
      it('should create fresh abort signal with specified timeout on retry', async () => {
        // Arrange
        let baseModelSignal: AbortSignal | undefined;
        let fallbackModelSignal: AbortSignal | undefined;
        const abortError = new DOMException(
          'The operation was aborted',
          'AbortError',
        );

        const baseModel = new MockLanguageModel({
          doGenerate: async (opts: LanguageModelCallOptions) => {
            baseModelSignal = opts.abortSignal;
            throw abortError;
          },
        });

        const fallbackModel = new MockLanguageModel({
          doGenerate: async (opts: LanguageModelCallOptions) => {
            fallbackModelSignal = opts.abortSignal;
            // Verify the new signal is not aborted
            if (opts.abortSignal?.aborted) {
              throw new Error('Should not be aborted with fresh signal');
            }
            return mockResult;
          },
        });

        // Create an already-aborted signal
        const controller = new AbortController();
        controller.abort();

        // Act
        await generateText({
          model: createRetryable({
            model: baseModel,
            retries: [
              {
                model: fallbackModel,
                timeout: 30000, // 30 second timeout for retry
              },
            ],
          }),
          prompt: 'Hello!',
          abortSignal: controller.signal,
        });

        // Assert
        expect(baseModel.doGenerate).toHaveBeenCalledTimes(1);
        expect(fallbackModel.doGenerate).toHaveBeenCalledTimes(1);

        // Base model should receive the original aborted signal
        expect(baseModelSignal?.aborted).toBe(true);

        // Fallback model should receive a fresh, non-aborted signal
        expect(fallbackModelSignal).toBeDefined();
        expect(fallbackModelSignal?.aborted).toBe(false);
        expect(baseModelSignal).not.toBe(fallbackModelSignal);
      });
    });

    describe('prompt', () => {
      it('should override prompt on retry', async () => {
        // Arrange
        const baseModel = new MockLanguageModel({ doGenerate: retryableError });
        const fallbackModel = new MockLanguageModel({ doGenerate: mockResult });
        const overridePrompt = [
          {
            role: 'user' as const,
            content: [{ type: 'text' as const, text: 'Modified prompt' }],
          },
        ];

        // Act
        await generateText({
          model: createRetryable({
            model: baseModel,
            retries: [
              { model: fallbackModel, options: { prompt: overridePrompt } },
            ],
          }),
          prompt: 'Original prompt',
        });

        // Assert
        expect(fallbackModel.doGenerate).toHaveBeenCalledWith(
          expect.objectContaining({ prompt: overridePrompt }),
        );
      });
    });

    describe('temperature', () => {
      it('should override temperature on retry', async () => {
        // Arrange
        const baseModel = new MockLanguageModel({ doGenerate: retryableError });
        const fallbackModel = new MockLanguageModel({ doGenerate: mockResult });

        // Act
        await generateText({
          model: createRetryable({
            model: baseModel,
            retries: [{ model: fallbackModel, options: { temperature: 0.5 } }],
          }),
          prompt: 'Hello!',
          temperature: 1.0,
        });

        // Assert
        expect(baseModel.doGenerate).toHaveBeenCalledWith(
          expect.objectContaining({ temperature: 1.0 }),
        );
        expect(fallbackModel.doGenerate).toHaveBeenCalledWith(
          expect.objectContaining({ temperature: 0.5 }),
        );
      });
    });

    describe('topP', () => {
      it('should override topP on retry', async () => {
        // Arrange
        const baseModel = new MockLanguageModel({ doGenerate: retryableError });
        const fallbackModel = new MockLanguageModel({ doGenerate: mockResult });

        // Act
        await generateText({
          model: createRetryable({
            model: baseModel,
            retries: [{ model: fallbackModel, options: { topP: 0.8 } }],
          }),
          prompt: 'Hello!',
          topP: 1.0,
        });

        // Assert
        expect(baseModel.doGenerate).toHaveBeenCalledWith(
          expect.objectContaining({ topP: 1.0 }),
        );
        expect(fallbackModel.doGenerate).toHaveBeenCalledWith(
          expect.objectContaining({ topP: 0.8 }),
        );
      });
    });

    describe('topK', () => {
      it('should override topK on retry', async () => {
        // Arrange
        const baseModel = new MockLanguageModel({ doGenerate: retryableError });
        const fallbackModel = new MockLanguageModel({ doGenerate: mockResult });

        // Act
        await generateText({
          model: createRetryable({
            model: baseModel,
            retries: [{ model: fallbackModel, options: { topK: 10 } }],
          }),
          prompt: 'Hello!',
          topK: 50,
        });

        // Assert
        expect(baseModel.doGenerate).toHaveBeenCalledWith(
          expect.objectContaining({ topK: 50 }),
        );
        expect(fallbackModel.doGenerate).toHaveBeenCalledWith(
          expect.objectContaining({ topK: 10 }),
        );
      });
    });

    describe('maxOutputTokens', () => {
      it('should override maxOutputTokens on retry', async () => {
        // Arrange
        const baseModel = new MockLanguageModel({ doGenerate: retryableError });
        const fallbackModel = new MockLanguageModel({ doGenerate: mockResult });

        // Act
        await generateText({
          model: createRetryable({
            model: baseModel,
            retries: [
              { model: fallbackModel, options: { maxOutputTokens: 500 } },
            ],
          }),
          prompt: 'Hello!',
          maxOutputTokens: 1000,
        });

        // Assert
        expect(baseModel.doGenerate).toHaveBeenCalledWith(
          expect.objectContaining({ maxOutputTokens: 1000 }),
        );
        expect(fallbackModel.doGenerate).toHaveBeenCalledWith(
          expect.objectContaining({ maxOutputTokens: 500 }),
        );
      });
    });

    describe('seed', () => {
      it('should override seed on retry', async () => {
        // Arrange
        const baseModel = new MockLanguageModel({ doGenerate: retryableError });
        const fallbackModel = new MockLanguageModel({ doGenerate: mockResult });

        // Act
        await generateText({
          model: createRetryable({
            model: baseModel,
            retries: [{ model: fallbackModel, options: { seed: 42 } }],
          }),
          prompt: 'Hello!',
          seed: 123,
        });

        // Assert
        expect(baseModel.doGenerate).toHaveBeenCalledWith(
          expect.objectContaining({ seed: 123 }),
        );
        expect(fallbackModel.doGenerate).toHaveBeenCalledWith(
          expect.objectContaining({ seed: 42 }),
        );
      });
    });

    describe('stopSequences', () => {
      it('should override stopSequences on retry', async () => {
        // Arrange
        const baseModel = new MockLanguageModel({ doGenerate: retryableError });
        const fallbackModel = new MockLanguageModel({ doGenerate: mockResult });

        // Act
        await generateText({
          model: createRetryable({
            model: baseModel,
            retries: [
              {
                model: fallbackModel,
                options: { stopSequences: ['RETRY_STOP'] },
              },
            ],
          }),
          prompt: 'Hello!',
          stopSequences: ['ORIGINAL_STOP'],
        });

        // Assert
        expect(baseModel.doGenerate).toHaveBeenCalledWith(
          expect.objectContaining({ stopSequences: ['ORIGINAL_STOP'] }),
        );
        expect(fallbackModel.doGenerate).toHaveBeenCalledWith(
          expect.objectContaining({ stopSequences: ['RETRY_STOP'] }),
        );
      });
    });

    describe('presencePenalty', () => {
      it('should override presencePenalty on retry', async () => {
        // Arrange
        const baseModel = new MockLanguageModel({ doGenerate: retryableError });
        const fallbackModel = new MockLanguageModel({ doGenerate: mockResult });

        // Act
        await generateText({
          model: createRetryable({
            model: baseModel,
            retries: [
              { model: fallbackModel, options: { presencePenalty: 0.5 } },
            ],
          }),
          prompt: 'Hello!',
          presencePenalty: 0.0,
        });

        // Assert
        expect(baseModel.doGenerate).toHaveBeenCalledWith(
          expect.objectContaining({ presencePenalty: 0.0 }),
        );
        expect(fallbackModel.doGenerate).toHaveBeenCalledWith(
          expect.objectContaining({ presencePenalty: 0.5 }),
        );
      });
    });

    describe('frequencyPenalty', () => {
      it('should override frequencyPenalty on retry', async () => {
        // Arrange
        const baseModel = new MockLanguageModel({ doGenerate: retryableError });
        const fallbackModel = new MockLanguageModel({ doGenerate: mockResult });

        // Act
        await generateText({
          model: createRetryable({
            model: baseModel,
            retries: [
              { model: fallbackModel, options: { frequencyPenalty: 0.8 } },
            ],
          }),
          prompt: 'Hello!',
          frequencyPenalty: 0.2,
        });

        // Assert
        expect(baseModel.doGenerate).toHaveBeenCalledWith(
          expect.objectContaining({ frequencyPenalty: 0.2 }),
        );
        expect(fallbackModel.doGenerate).toHaveBeenCalledWith(
          expect.objectContaining({ frequencyPenalty: 0.8 }),
        );
      });
    });

    describe('headers', () => {
      it('should override headers on retry', async () => {
        // Arrange
        const baseModel = new MockLanguageModel({ doGenerate: retryableError });
        const fallbackModel = new MockLanguageModel({ doGenerate: mockResult });
        // Use lowercase headers to match what AI SDK expects
        const retryHeaders = { 'x-retry': 'retry' };

        // Act
        await generateText({
          model: createRetryable({
            model: baseModel,
            retries: [
              { model: fallbackModel, options: { headers: retryHeaders } },
            ],
          }),
          prompt: 'Hello!',
        });

        // Assert - check that retry headers are passed to fallback model
        expect(fallbackModel.doGenerate).toHaveBeenCalledWith(
          expect.objectContaining({
            headers: expect.objectContaining({
              'x-retry': 'retry',
            }),
          }),
        );
      });
    });
  });

  describe('RetryError', () => {
    it('should throw RetryError when all retry attempts are exhausted', async () => {
      // Arrange
      const baseModel = new MockLanguageModel({ doGenerate: retryableError });
      const fallbackModel1 = new MockLanguageModel({
        doGenerate: nonRetryableError,
      });
      const fallbackModel2 = new MockLanguageModel({
        doGenerate: retryableError,
      });

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
      const baseModel = new MockLanguageModel({ doGenerate: retryableError });

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
      const baseModel = new MockLanguageModel({
        doGenerate: nonRetryableError,
      });

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
});

describe('streamText', () => {
  it('should stream successfully when no errors occur', async () => {
    const baseModel = new MockLanguageModel({
      doStream: {
        stream: convertArrayToReadableStream(mockStreamChunks),
      },
    });

    const retryableModel = createRetryable({
      model: baseModel,
      retries: [],
    });

    const result = streamText({
      model: retryableModel,
      prompt,
    });

    const chunks = await convertAsyncIterableToArray(result.fullStream);

    expect(baseModel.doStream).toHaveBeenCalledTimes(1);
    expect(chunksToText(chunks)).toBe('Hello, world!');
  });

  describe('retries', () => {
    it('should retry when error occurs at stream creation', async () => {
      const baseModel = new MockLanguageModel({ doStream: retryableError });

      const fallbackModel = new MockLanguageModel({
        doStream: {
          stream: convertArrayToReadableStream(mockStreamChunks),
        },
      });

      const retryableModel = createRetryable({
        model: baseModel,
        retries: [fallbackModel],
      });

      const result = streamText({
        model: retryableModel,
        prompt,
      });

      const chunks = await convertAsyncIterableToArray(result.fullStream);

      expect(baseModel.doStream).toHaveBeenCalledTimes(1);
      expect(fallbackModel.doStream).toHaveBeenCalledTimes(1);
      expect(chunks).toMatchInlineSnapshot(`
        [
          {
            "type": "start",
          },
          {
            "request": {},
            "type": "start-step",
            "warnings": [],
          },
          {
            "id": "1",
            "type": "text-start",
          },
          {
            "id": "1",
            "providerMetadata": undefined,
            "text": "Hello",
            "type": "text-delta",
          },
          {
            "id": "1",
            "providerMetadata": undefined,
            "text": ", ",
            "type": "text-delta",
          },
          {
            "id": "1",
            "providerMetadata": undefined,
            "text": "world!",
            "type": "text-delta",
          },
          {
            "id": "1",
            "type": "text-end",
          },
          {
            "finishReason": "stop",
            "providerMetadata": {
              "testProvider": {
                "testKey": "testValue",
              },
            },
            "response": {
              "headers": undefined,
              "id": "id-0",
              "modelId": "mock-model-id",
              "timestamp": 1970-01-01T00:00:00.000Z,
            },
            "type": "finish-step",
            "usage": {
              "cachedInputTokens": 0,
              "inputTokenDetails": {
                "cacheReadTokens": 0,
                "cacheWriteTokens": 0,
                "noCacheTokens": 0,
              },
              "inputTokens": 3,
              "outputTokenDetails": {
                "reasoningTokens": 0,
                "textTokens": 0,
              },
              "outputTokens": 10,
              "raw": undefined,
              "reasoningTokens": 0,
              "totalTokens": 13,
            },
          },
          {
            "finishReason": "stop",
            "totalUsage": {
              "cachedInputTokens": 0,
              "inputTokenDetails": {
                "cacheReadTokens": 0,
                "cacheWriteTokens": 0,
                "noCacheTokens": 0,
              },
              "inputTokens": 3,
              "outputTokenDetails": {
                "reasoningTokens": 0,
                "textTokens": 0,
              },
              "outputTokens": 10,
              "reasoningTokens": 0,
              "totalTokens": 13,
            },
            "type": "finish",
          },
        ]
      `);
    });

    it('should retry when error occurs at the stream start', async () => {
      const baseModel = new MockLanguageModel({
        doStream: {
          stream: convertArrayToReadableStream([
            { type: 'stream-start', warnings: [] },
            {
              type: 'error',
              error: { type: 'overloaded_error', message: 'Overloaded' },
            },
          ]),
        },
      });

      const fallbackModel = new MockLanguageModel({
        doStream: {
          stream: convertArrayToReadableStream(mockStreamChunks),
        },
      });

      const retryableModel = createRetryable({
        model: baseModel,
        retries: [fallbackModel],
      });

      const result = streamText({
        model: retryableModel,
        prompt,
      });

      const chunks = await convertAsyncIterableToArray(result.fullStream);

      expect(baseModel.doStream).toHaveBeenCalledTimes(1);
      expect(fallbackModel.doStream).toHaveBeenCalledTimes(1);
      expect(chunks).toMatchInlineSnapshot(`
        [
          {
            "type": "start",
          },
          {
            "request": {},
            "type": "start-step",
            "warnings": [],
          },
          {
            "id": "1",
            "type": "text-start",
          },
          {
            "id": "1",
            "providerMetadata": undefined,
            "text": "Hello",
            "type": "text-delta",
          },
          {
            "id": "1",
            "providerMetadata": undefined,
            "text": ", ",
            "type": "text-delta",
          },
          {
            "id": "1",
            "providerMetadata": undefined,
            "text": "world!",
            "type": "text-delta",
          },
          {
            "id": "1",
            "type": "text-end",
          },
          {
            "finishReason": "stop",
            "providerMetadata": {
              "testProvider": {
                "testKey": "testValue",
              },
            },
            "response": {
              "headers": undefined,
              "id": "id-0",
              "modelId": "mock-model-id",
              "timestamp": 1970-01-01T00:00:00.000Z,
            },
            "type": "finish-step",
            "usage": {
              "cachedInputTokens": 0,
              "inputTokenDetails": {
                "cacheReadTokens": 0,
                "cacheWriteTokens": 0,
                "noCacheTokens": 0,
              },
              "inputTokens": 3,
              "outputTokenDetails": {
                "reasoningTokens": 0,
                "textTokens": 0,
              },
              "outputTokens": 10,
              "raw": undefined,
              "reasoningTokens": 0,
              "totalTokens": 13,
            },
          },
          {
            "finishReason": "stop",
            "totalUsage": {
              "cachedInputTokens": 0,
              "inputTokenDetails": {
                "cacheReadTokens": 0,
                "cacheWriteTokens": 0,
                "noCacheTokens": 0,
              },
              "inputTokens": 3,
              "outputTokenDetails": {
                "reasoningTokens": 0,
                "textTokens": 0,
              },
              "outputTokens": 10,
              "reasoningTokens": 0,
              "totalTokens": 13,
            },
            "type": "finish",
          },
        ]
      `);
    });

    it('should retry when consective errors occur', async () => {
      const baseModel = new MockLanguageModel({
        doStream: {
          stream: convertArrayToReadableStream([
            { type: 'stream-start', warnings: [] },
            {
              type: 'error',
              error: { type: 'overloaded_error', message: 'Overloaded' },
            },
          ]),
        },
      });

      const fallbackModel1 = new MockLanguageModel({
        doStream: retryableError,
      });
      const fallbackModel2 = new MockLanguageModel({
        doStream: {
          stream: convertArrayToReadableStream(mockStreamChunks),
        },
      });

      const retryableModel = createRetryable({
        model: baseModel,
        retries: [fallbackModel1, fallbackModel2],
      });

      const result = streamText({
        model: retryableModel,
        prompt,
      });

      const chunks = await convertAsyncIterableToArray(result.fullStream);

      expect(baseModel.doStream).toHaveBeenCalledTimes(1);
      expect(fallbackModel1.doStream).toHaveBeenCalledTimes(1);
      expect(fallbackModel2.doStream).toHaveBeenCalledTimes(1);
      expect(chunks).toMatchInlineSnapshot(`
        [
          {
            "type": "start",
          },
          {
            "request": {},
            "type": "start-step",
            "warnings": [],
          },
          {
            "id": "1",
            "type": "text-start",
          },
          {
            "id": "1",
            "providerMetadata": undefined,
            "text": "Hello",
            "type": "text-delta",
          },
          {
            "id": "1",
            "providerMetadata": undefined,
            "text": ", ",
            "type": "text-delta",
          },
          {
            "id": "1",
            "providerMetadata": undefined,
            "text": "world!",
            "type": "text-delta",
          },
          {
            "id": "1",
            "type": "text-end",
          },
          {
            "finishReason": "stop",
            "providerMetadata": {
              "testProvider": {
                "testKey": "testValue",
              },
            },
            "response": {
              "headers": undefined,
              "id": "id-0",
              "modelId": "mock-model-id",
              "timestamp": 1970-01-01T00:00:00.000Z,
            },
            "type": "finish-step",
            "usage": {
              "cachedInputTokens": 0,
              "inputTokenDetails": {
                "cacheReadTokens": 0,
                "cacheWriteTokens": 0,
                "noCacheTokens": 0,
              },
              "inputTokens": 3,
              "outputTokenDetails": {
                "reasoningTokens": 0,
                "textTokens": 0,
              },
              "outputTokens": 10,
              "raw": undefined,
              "reasoningTokens": 0,
              "totalTokens": 13,
            },
          },
          {
            "finishReason": "stop",
            "totalUsage": {
              "cachedInputTokens": 0,
              "inputTokenDetails": {
                "cacheReadTokens": 0,
                "cacheWriteTokens": 0,
                "noCacheTokens": 0,
              },
              "inputTokens": 3,
              "outputTokenDetails": {
                "reasoningTokens": 0,
                "textTokens": 0,
              },
              "outputTokens": 10,
              "reasoningTokens": 0,
              "totalTokens": 13,
            },
            "type": "finish",
          },
        ]
      `);
    });

    it('should NOT retry when error occurs during streaming', async () => {
      const baseModel = new MockLanguageModel({
        doStream: {
          stream: convertArrayToReadableStream([
            { type: 'stream-start', warnings: [] },
            { type: 'text-start', id: '1' },
            { type: 'text-delta', id: '1', delta: 'Hello' },
            { type: 'error', error: new Error('Overloaded') },
          ]),
        },
      });

      const fallbackModel = new MockLanguageModel({
        doStream: {
          stream: convertArrayToReadableStream(mockStreamChunks),
        },
      });

      const retryableModel = createRetryable({
        model: baseModel,
        retries: [fallbackModel],
      });

      const result = streamText({
        model: retryableModel,
        prompt,
        ...mockStreamOptions,
      });

      const chunks = await convertAsyncIterableToArray(result.fullStream);

      expect(baseModel.doStream).toHaveBeenCalledTimes(1);
      expect(fallbackModel.doStream).toHaveBeenCalledTimes(0);
      expect(chunks).toMatchInlineSnapshot(`
        [
          {
            "type": "start",
          },
          {
            "request": {},
            "type": "start-step",
            "warnings": [],
          },
          {
            "id": "1",
            "type": "text-start",
          },
          {
            "id": "1",
            "providerMetadata": undefined,
            "text": "Hello",
            "type": "text-delta",
          },
          {
            "error": [Error: Overloaded],
            "type": "error",
          },
          {
            "finishReason": "error",
            "providerMetadata": undefined,
            "response": {
              "headers": undefined,
              "id": "aitxt-mock-id",
              "modelId": "mock-model-112",
              "timestamp": 1970-01-01T00:00:00.000Z,
            },
            "type": "finish-step",
            "usage": {
              "inputTokenDetails": {
                "cacheReadTokens": undefined,
                "cacheWriteTokens": undefined,
                "noCacheTokens": undefined,
              },
              "inputTokens": undefined,
              "outputTokenDetails": {
                "reasoningTokens": undefined,
                "textTokens": undefined,
              },
              "outputTokens": undefined,
              "raw": undefined,
              "totalTokens": undefined,
            },
          },
          {
            "finishReason": "error",
            "totalUsage": {
              "cachedInputTokens": undefined,
              "inputTokenDetails": {
                "cacheReadTokens": undefined,
                "cacheWriteTokens": undefined,
                "noCacheTokens": undefined,
              },
              "inputTokens": undefined,
              "outputTokenDetails": {
                "reasoningTokens": undefined,
                "textTokens": undefined,
              },
              "outputTokens": undefined,
              "reasoningTokens": undefined,
              "totalTokens": undefined,
            },
            "type": "finish",
          },
        ]
      `);
    });
  });

  describe('disabled', () => {
    it('should not retry when disabled is true', async () => {
      // Arrange
      const baseModel = new MockLanguageModel({ doStream: retryableError });
      const fallbackModel = new MockLanguageModel({
        doStream: {
          stream: convertArrayToReadableStream(mockStreamChunks),
        },
      });

      const fallbackRetryable: Retryable<LanguageModel> = () => {
        return { model: fallbackModel, maxAttempts: 1 };
      };

      const retryableModel = createRetryable({
        model: baseModel,
        retries: [fallbackRetryable],
        disabled: true,
      });

      // Act
      const result = streamText({
        model: retryableModel,
        prompt: 'Hello!',
        maxRetries: 0, // Disable AI SDK's own retry mechanism
      });

      const chunks = await convertAsyncIterableToArray(result.fullStream);

      // Assert
      expect(baseModel.doStream).toHaveBeenCalledTimes(1);
      expect(fallbackModel.doStream).toHaveBeenCalledTimes(0);

      // Check that an error chunk was emitted
      const errorChunk: any = chunks.find(
        (chunk: any) => chunk.type === 'error',
      );
      expect(errorChunk).toBeDefined();
      expect(errorChunk.error.message).toBe('Rate limit exceeded');
    });

    it('should retry when disabled is false', async () => {
      // Arrange
      const baseModel = new MockLanguageModel({ doStream: retryableError });
      const fallbackModel = new MockLanguageModel({
        doStream: {
          stream: convertArrayToReadableStream(mockStreamChunks),
        },
      });

      const fallbackRetryable: Retryable<LanguageModel> = () => {
        return { model: fallbackModel, maxAttempts: 1 };
      };

      const retryableModel = createRetryable({
        model: baseModel,
        retries: [fallbackRetryable],
        disabled: false,
      });

      // Act
      const result = streamText({
        model: retryableModel,
        prompt: 'Hello!',
      });

      const chunks = await convertAsyncIterableToArray(result.textStream);

      // Assert
      expect(baseModel.doStream).toHaveBeenCalledTimes(1);
      expect(fallbackModel.doStream).toHaveBeenCalledTimes(1);
      expect(chunks.join('')).toBe('Hello, world!');
    });
  });

  describe('onError', () => {
    it('should call onError handler when an error occurs', async () => {
      // Arrange
      const baseModel = new MockLanguageModel({ doStream: retryableError });
      const fallbackModel = new MockLanguageModel({
        doStream: {
          stream: convertArrayToReadableStream(mockStreamChunks),
        },
      });
      const onErrorSpy = vi.fn<OnError>();

      // Act
      const retryableModel = createRetryable({
        model: baseModel,
        retries: [fallbackModel],
        onError: onErrorSpy,
      });

      const result = streamText({
        model: retryableModel,
        prompt,
      });

      await convertAsyncIterableToArray(result.fullStream);

      // Assert
      expect(onErrorSpy).toHaveBeenCalledTimes(1);

      const firstErrorCall = onErrorSpy.mock.calls[0]![0];
      expect(firstErrorCall.current.error).toBe(retryableError);
      expect(firstErrorCall.current.model).toBe(baseModel);
      expect(firstErrorCall.attempts.length).toBe(1);
      // expect(firstErrorCall.totalAttempts).toBe(1);
    });

    it('should call onError handler for each error in multiple attempts', async () => {
      // Arrange
      const baseModel = new MockLanguageModel({ doStream: retryableError });
      const fallbackModel = new MockLanguageModel({
        doStream: nonRetryableError,
      });
      const finalModel = new MockLanguageModel({
        doStream: {
          stream: convertArrayToReadableStream(mockStreamChunks),
        },
      });
      const onErrorSpy = vi.fn<OnError>();

      // Act
      const retryableModel = createRetryable({
        model: baseModel,
        retries: [fallbackModel, finalModel],
        onError: onErrorSpy,
      });

      const result = streamText({
        model: retryableModel,
        prompt,
      });

      await convertAsyncIterableToArray(result.fullStream);

      // Assert
      expect(onErrorSpy).toHaveBeenCalledTimes(2);

      // Check that onError was called for each error
      const firstErrorCall = onErrorSpy.mock.calls[0]![0];
      const secondErrorCall = onErrorSpy.mock.calls[1]![0];

      expect(firstErrorCall.current.error).toBe(retryableError);
      expect(firstErrorCall.current.model).toBe(baseModel);
      expect(firstErrorCall.attempts.length).toBe(1);
      // expect(firstErrorCall.totalAttempts).toBe(1);

      expect(secondErrorCall.current.error).toBe(nonRetryableError);
      expect(secondErrorCall.current.model).toBe(fallbackModel);
      expect(secondErrorCall.attempts.length).toBe(2);
      // expect(secondErrorCall.totalAttempts).toBe(2);
    });

    it('should NOT call onError handler when streaming succeeds', async () => {
      // Arrange
      const baseModel = new MockLanguageModel({
        doStream: {
          stream: convertArrayToReadableStream(mockStreamChunks),
        },
      });
      const onErrorSpy = vi.fn<OnError>();

      // Act
      const retryableModel = createRetryable({
        model: baseModel,
        retries: [],
        onError: onErrorSpy,
      });

      const result = streamText({
        model: retryableModel,
        prompt,
      });

      await convertAsyncIterableToArray(result.fullStream);

      // Assert
      expect(onErrorSpy).not.toHaveBeenCalled();
    });
  });

  describe('onRetry', () => {
    it('should call onRetry handler for error-based retries', async () => {
      // Arrange
      const baseModel = new MockLanguageModel({ doStream: retryableError });
      const fallbackModel = new MockLanguageModel({
        doStream: {
          stream: convertArrayToReadableStream(mockStreamChunks),
        },
      });
      const onRetrySpy = vi.fn<OnRetry>();

      // Act
      const retryableModel = createRetryable({
        model: baseModel,
        retries: [fallbackModel],
        onRetry: onRetrySpy,
      });

      const result = streamText({
        model: retryableModel,
        prompt,
      });

      await convertAsyncIterableToArray(result.fullStream);

      // Assert
      expect(onRetrySpy).toHaveBeenCalledTimes(1);

      const firstRetryCall = onRetrySpy.mock.calls[0]![0];
      expect(isErrorAttempt(firstRetryCall.current)).toBe(true);
      if (isErrorAttempt(firstRetryCall.current)) {
        expect(firstRetryCall.current.error).toBe(retryableError);
      }
      expect(firstRetryCall.current.model).toBe(fallbackModel);
      expect(firstRetryCall.attempts.length).toBe(1);
      // expect(firstRetryCall.totalAttempts).toBe(1);
    });

    it('should call onRetry handler for each retry attempt', async () => {
      // Arrange
      const baseModel = new MockLanguageModel({ doStream: retryableError });
      const fallbackModel1 = new MockLanguageModel({
        doStream: nonRetryableError,
      });
      const fallbackModel2 = new MockLanguageModel({
        doStream: {
          stream: convertArrayToReadableStream(mockStreamChunks),
        },
      });
      const onRetrySpy = vi.fn<OnRetry>();

      // Act
      const retryableModel = createRetryable({
        model: baseModel,
        retries: [fallbackModel1, fallbackModel2],
        onRetry: onRetrySpy,
      });

      const result = streamText({
        model: retryableModel,
        prompt,
      });

      await convertAsyncIterableToArray(result.fullStream);

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
      expect(firstRetryCall.attempts.length).toBe(1);
      // expect(firstRetryCall.totalAttempts).toBe(1);

      expect(isErrorAttempt(secondRetryCall.current)).toBe(true);
      if (isErrorAttempt(secondRetryCall.current)) {
        expect(secondRetryCall.current.error).toBe(nonRetryableError);
      }
      expect(secondRetryCall.current.model).toBe(fallbackModel2);
      expect(secondRetryCall.attempts.length).toBe(2);
      // expect(secondRetryCall.totalAttempts).toBe(2);
    });

    it('should NOT call onRetry on first attempt', async () => {
      // Arrange
      const baseModel = new MockLanguageModel({
        doStream: {
          stream: convertArrayToReadableStream(mockStreamChunks),
        },
      });
      const onRetrySpy = vi.fn<OnRetry>();

      // Act
      const retryableModel = createRetryable({
        model: baseModel,
        retries: [],
        onRetry: onRetrySpy,
      });

      const result = streamText({
        model: retryableModel,
        prompt,
      });

      await convertAsyncIterableToArray(result.fullStream);

      // Assert
      expect(onRetrySpy).not.toHaveBeenCalled();
    });
  });

  describe('RetryableOptions', () => {
    describe('maxAttempts', () => {
      it('should try each model only once by default', async () => {
        // Arrange
        const baseModel = new MockLanguageModel({ doStream: retryableError });
        const fallbackModel1 = new MockLanguageModel({
          doStream: retryableError,
        });
        const fallbackModel2 = new MockLanguageModel({
          doStream: retryableError,
        });
        const finalModel = new MockLanguageModel({
          doStream: {
            stream: convertArrayToReadableStream(mockStreamChunks),
          },
        });

        // Act
        const retryableModel = createRetryable({
          model: baseModel,
          retries: [
            fallbackModel1,
            { model: fallbackModel2 },
            async () => ({ model: finalModel }),
          ],
        });

        const result = streamText({
          model: retryableModel,
          prompt,
        });

        await convertAsyncIterableToArray(result.fullStream);

        // Assert
        expect(baseModel.doStream).toHaveBeenCalledTimes(1);
        expect(fallbackModel1.doStream).toHaveBeenCalledTimes(1);
        expect(fallbackModel2.doStream).toHaveBeenCalledTimes(1);
        expect(finalModel.doStream).toHaveBeenCalledTimes(1);
      });

      it('should try models multiple times if maxAttempts is set', async () => {
        // Arrange
        const baseModel = new MockLanguageModel({ doStream: retryableError });
        const fallbackModel1 = new MockLanguageModel({
          doStream: retryableError,
        });
        const fallbackModel2 = new MockLanguageModel({
          doStream: retryableError,
        });
        const finalModel = new MockLanguageModel({
          doStream: {
            stream: convertArrayToReadableStream(mockStreamChunks),
          },
        });

        // Act
        const retryableModel = createRetryable({
          model: baseModel,
          retries: [
            // Retryable<LanguageModel>  with different maxAttempts
            { model: fallbackModel1, maxAttempts: 2 },
            async () => ({ model: fallbackModel2, maxAttempts: 3 }),
            finalModel,
          ],
        });

        const result = streamText({
          model: retryableModel,
          prompt,
        });

        await convertAsyncIterableToArray(result.fullStream);

        // Assert
        expect(baseModel.doStream).toHaveBeenCalledTimes(1);
        expect(fallbackModel1.doStream).toHaveBeenCalledTimes(2);
        expect(fallbackModel2.doStream).toHaveBeenCalledTimes(3);
        expect(finalModel.doStream).toHaveBeenCalledTimes(1);
      });
    });

    describe('delay', () => {
      it('should apply delay before retrying', async () => {
        // Arrange
        vi.useFakeTimers();
        const baseModel = new MockLanguageModel({ doStream: retryableError });
        const fallbackModel = new MockLanguageModel({
          doStream: {
            stream: convertArrayToReadableStream(mockStreamChunks),
          },
        });
        const delayMs = 100;

        // Act
        const result = streamText({
          model: createRetryable({
            model: baseModel,
            retries: [{ model: fallbackModel, delay: delayMs }],
          }),
          prompt,
        });

        const streamPromise = convertAsyncIterableToArray(result.fullStream);
        await vi.runAllTimersAsync();
        await streamPromise;

        // Assert
        expect(baseModel.doStream).toHaveBeenCalledTimes(1);
        expect(fallbackModel.doStream).toHaveBeenCalledTimes(1);

        vi.useRealTimers();
      });

      it('should apply different delays for multiple retries', async () => {
        // Arrange
        vi.useFakeTimers();
        const baseModel = new MockLanguageModel({ doStream: retryableError });
        const fallbackModel1 = new MockLanguageModel({
          doStream: retryableError,
        });
        const fallbackModel2 = new MockLanguageModel({
          doStream: {
            stream: convertArrayToReadableStream(mockStreamChunks),
          },
        });
        const delay1 = 50;
        const delay2 = 50;

        // Act
        const result = streamText({
          model: createRetryable({
            model: baseModel,
            retries: [
              { model: fallbackModel1, delay: delay1 },
              () => ({ model: fallbackModel2, delay: delay2 }),
            ],
          }),
          prompt,
        });

        const streamPromise = convertAsyncIterableToArray(result.fullStream);
        await vi.runAllTimersAsync();
        await streamPromise;

        // Assert
        expect(baseModel.doStream).toHaveBeenCalledTimes(1);
        expect(fallbackModel1.doStream).toHaveBeenCalledTimes(1);
        expect(fallbackModel2.doStream).toHaveBeenCalledTimes(1);

        vi.useRealTimers();
      });

      it('should not delay when delay is not specified', async () => {
        // Arrange
        const baseModel = new MockLanguageModel({ doStream: retryableError });
        const fallbackModel = new MockLanguageModel({
          doStream: {
            stream: convertArrayToReadableStream(mockStreamChunks),
          },
        });

        // Act
        const result = streamText({
          model: createRetryable({
            model: baseModel,
            retries: [{ model: fallbackModel }],
          }),
          prompt,
        });

        await convertAsyncIterableToArray(result.fullStream);

        // Assert
        expect(baseModel.doStream).toHaveBeenCalledTimes(1);
        expect(fallbackModel.doStream).toHaveBeenCalledTimes(1);
      });
    });

    describe('providerOptions', () => {
      it('should override base model providerOptions with retry model providerOptions', async () => {
        // Arrange
        const baseModel = new MockLanguageModel({ doStream: retryableError });
        const fallbackModel = new MockLanguageModel({
          doStream: {
            stream: convertArrayToReadableStream(mockStreamChunks),
          },
        });
        const originalProviderOptions = {
          openai: { user: 'original-user' },
        };
        const retryProviderOptions = { openai: { user: 'retry-user' } };

        // Act
        const result = streamText({
          model: createRetryable({
            model: baseModel,
            retries: [
              {
                model: fallbackModel,
                providerOptions: retryProviderOptions,
              },
            ],
          }),
          prompt,
          providerOptions: originalProviderOptions,
        });

        await convertAsyncIterableToArray(result.fullStream);

        // Assert
        expect(baseModel.doStream).toHaveBeenCalledTimes(1);
        expect(fallbackModel.doStream).toHaveBeenCalledTimes(1);
        expect(baseModel.doStream).toHaveBeenCalledWith(
          expect.objectContaining({
            providerOptions: originalProviderOptions,
          }),
        );
        expect(fallbackModel.doStream).toHaveBeenCalledWith(
          expect.objectContaining({
            providerOptions: retryProviderOptions,
          }),
        );
      });
    });

    describe('timeout', () => {
      it('should create fresh abort signal with specified timeout on retry', async () => {
        // Arrange
        let baseModelSignal: AbortSignal | undefined;
        let fallbackModelSignal: AbortSignal | undefined;
        const abortError = new DOMException(
          'The operation was aborted',
          'AbortError',
        );

        const baseModel = new MockLanguageModel({
          doStream: async (opts: LanguageModelCallOptions) => {
            baseModelSignal = opts.abortSignal;
            throw abortError;
          },
        });

        const fallbackModel = new MockLanguageModel({
          doStream: async (opts: LanguageModelCallOptions) => {
            fallbackModelSignal = opts.abortSignal;
            // Verify the new signal is not aborted
            if (opts.abortSignal?.aborted) {
              throw new Error('Should not be aborted with fresh signal');
            }
            return {
              stream: convertArrayToReadableStream(mockStreamChunks),
            };
          },
        });

        // Create an already-aborted signal
        const controller = new AbortController();
        controller.abort();

        // Act
        const result = streamText({
          model: createRetryable({
            model: baseModel,
            retries: [
              {
                model: fallbackModel,
                timeout: 30000, // 30 second timeout for retry
              },
            ],
          }),
          prompt,
          abortSignal: controller.signal,
        });

        await convertAsyncIterableToArray(result.fullStream);

        // Assert
        expect(baseModel.doStream).toHaveBeenCalledTimes(1);
        expect(fallbackModel.doStream).toHaveBeenCalledTimes(1);

        // Base model should receive the original aborted signal
        expect(baseModelSignal?.aborted).toBe(true);

        // Fallback model should receive a fresh, non-aborted signal
        expect(fallbackModelSignal).toBeDefined();
        expect(fallbackModelSignal?.aborted).toBe(false);
        expect(baseModelSignal).not.toBe(fallbackModelSignal);
      });
    });

    describe('temperature', () => {
      it('should override temperature on retry after stream error', async () => {
        // Arrange
        // Base model returns a stream that errors before content starts
        const baseModel = new MockLanguageModel({
          doStream: {
            stream: convertArrayToReadableStream([
              { type: 'stream-start', warnings: [] },
              { type: 'error', error: new Error('Stream error') },
            ]),
          },
        });

        const fallbackModel = new MockLanguageModel({
          doStream: {
            stream: convertArrayToReadableStream(mockStreamChunks),
          },
        });

        const retryableModel = createRetryable({
          model: baseModel,
          retries: [{ model: fallbackModel, options: { temperature: 0.5 } }],
        });

        // Act
        const result = streamText({
          model: retryableModel,
          prompt: 'Hello!',
          temperature: 1.0,
        });

        await convertAsyncIterableToArray(result.fullStream);

        // Assert
        // Base model should be called with original temperature
        expect(baseModel.doStream).toHaveBeenCalledWith(
          expect.objectContaining({ temperature: 1.0 }),
        );

        // Fallback model should be called with overridden temperature
        expect(fallbackModel.doStream).toHaveBeenCalledWith(
          expect.objectContaining({ temperature: 0.5 }),
        );
      });
    });

    describe('topP', () => {
      it('should override topP on retry after stream error', async () => {
        // Arrange
        const baseModel = new MockLanguageModel({
          doStream: {
            stream: convertArrayToReadableStream([
              { type: 'stream-start', warnings: [] },
              { type: 'error', error: new Error('Stream error') },
            ]),
          },
        });

        const fallbackModel = new MockLanguageModel({
          doStream: {
            stream: convertArrayToReadableStream(mockStreamChunks),
          },
        });

        const retryableModel = createRetryable({
          model: baseModel,
          retries: [{ model: fallbackModel, options: { topP: 0.8 } }],
        });

        // Act
        const result = streamText({
          model: retryableModel,
          prompt: 'Hello!',
          topP: 1.0,
        });

        await convertAsyncIterableToArray(result.fullStream);

        // Assert
        expect(baseModel.doStream).toHaveBeenCalledWith(
          expect.objectContaining({ topP: 1.0 }),
        );
        expect(fallbackModel.doStream).toHaveBeenCalledWith(
          expect.objectContaining({ topP: 0.8 }),
        );
      });
    });

    describe('topK', () => {
      it('should override topK on retry after stream error', async () => {
        // Arrange
        const baseModel = new MockLanguageModel({
          doStream: {
            stream: convertArrayToReadableStream([
              { type: 'stream-start', warnings: [] },
              { type: 'error', error: new Error('Stream error') },
            ]),
          },
        });

        const fallbackModel = new MockLanguageModel({
          doStream: {
            stream: convertArrayToReadableStream(mockStreamChunks),
          },
        });

        const retryableModel = createRetryable({
          model: baseModel,
          retries: [{ model: fallbackModel, options: { topK: 10 } }],
        });

        // Act
        const result = streamText({
          model: retryableModel,
          prompt: 'Hello!',
          topK: 50,
        });

        await convertAsyncIterableToArray(result.fullStream);

        // Assert
        expect(baseModel.doStream).toHaveBeenCalledWith(
          expect.objectContaining({ topK: 50 }),
        );
        expect(fallbackModel.doStream).toHaveBeenCalledWith(
          expect.objectContaining({ topK: 10 }),
        );
      });
    });

    describe('maxOutputTokens', () => {
      it('should override maxOutputTokens on retry after stream error', async () => {
        // Arrange
        const baseModel = new MockLanguageModel({
          doStream: {
            stream: convertArrayToReadableStream([
              { type: 'stream-start', warnings: [] },
              { type: 'error', error: new Error('Stream error') },
            ]),
          },
        });

        const fallbackModel = new MockLanguageModel({
          doStream: {
            stream: convertArrayToReadableStream(mockStreamChunks),
          },
        });

        const retryableModel = createRetryable({
          model: baseModel,
          retries: [
            { model: fallbackModel, options: { maxOutputTokens: 500 } },
          ],
        });

        // Act
        const result = streamText({
          model: retryableModel,
          prompt: 'Hello!',
          maxOutputTokens: 1000,
        });

        await convertAsyncIterableToArray(result.fullStream);

        // Assert
        expect(baseModel.doStream).toHaveBeenCalledWith(
          expect.objectContaining({ maxOutputTokens: 1000 }),
        );
        expect(fallbackModel.doStream).toHaveBeenCalledWith(
          expect.objectContaining({ maxOutputTokens: 500 }),
        );
      });
    });

    describe('seed', () => {
      it('should override seed on retry after stream error', async () => {
        // Arrange
        const baseModel = new MockLanguageModel({
          doStream: {
            stream: convertArrayToReadableStream([
              { type: 'stream-start', warnings: [] },
              { type: 'error', error: new Error('Stream error') },
            ]),
          },
        });

        const fallbackModel = new MockLanguageModel({
          doStream: {
            stream: convertArrayToReadableStream(mockStreamChunks),
          },
        });

        const retryableModel = createRetryable({
          model: baseModel,
          retries: [{ model: fallbackModel, options: { seed: 42 } }],
        });

        // Act
        const result = streamText({
          model: retryableModel,
          prompt: 'Hello!',
          seed: 123,
        });

        await convertAsyncIterableToArray(result.fullStream);

        // Assert
        expect(baseModel.doStream).toHaveBeenCalledWith(
          expect.objectContaining({ seed: 123 }),
        );
        expect(fallbackModel.doStream).toHaveBeenCalledWith(
          expect.objectContaining({ seed: 42 }),
        );
      });
    });

    describe('stopSequences', () => {
      it('should override stopSequences on retry after stream error', async () => {
        // Arrange
        const baseModel = new MockLanguageModel({
          doStream: {
            stream: convertArrayToReadableStream([
              { type: 'stream-start', warnings: [] },
              { type: 'error', error: new Error('Stream error') },
            ]),
          },
        });

        const fallbackModel = new MockLanguageModel({
          doStream: {
            stream: convertArrayToReadableStream(mockStreamChunks),
          },
        });

        const retryableModel = createRetryable({
          model: baseModel,
          retries: [
            {
              model: fallbackModel,
              options: { stopSequences: ['RETRY_STOP'] },
            },
          ],
        });

        // Act
        const result = streamText({
          model: retryableModel,
          prompt: 'Hello!',
          stopSequences: ['ORIGINAL_STOP'],
        });

        await convertAsyncIterableToArray(result.fullStream);

        // Assert
        expect(baseModel.doStream).toHaveBeenCalledWith(
          expect.objectContaining({ stopSequences: ['ORIGINAL_STOP'] }),
        );
        expect(fallbackModel.doStream).toHaveBeenCalledWith(
          expect.objectContaining({ stopSequences: ['RETRY_STOP'] }),
        );
      });
    });

    describe('presencePenalty', () => {
      it('should override presencePenalty on retry after stream error', async () => {
        // Arrange
        const baseModel = new MockLanguageModel({
          doStream: {
            stream: convertArrayToReadableStream([
              { type: 'stream-start', warnings: [] },
              { type: 'error', error: new Error('Stream error') },
            ]),
          },
        });

        const fallbackModel = new MockLanguageModel({
          doStream: {
            stream: convertArrayToReadableStream(mockStreamChunks),
          },
        });

        const retryableModel = createRetryable({
          model: baseModel,
          retries: [
            { model: fallbackModel, options: { presencePenalty: 0.5 } },
          ],
        });

        // Act
        const result = streamText({
          model: retryableModel,
          prompt: 'Hello!',
          presencePenalty: 0.0,
        });

        await convertAsyncIterableToArray(result.fullStream);

        // Assert
        expect(baseModel.doStream).toHaveBeenCalledWith(
          expect.objectContaining({ presencePenalty: 0.0 }),
        );
        expect(fallbackModel.doStream).toHaveBeenCalledWith(
          expect.objectContaining({ presencePenalty: 0.5 }),
        );
      });
    });

    describe('frequencyPenalty', () => {
      it('should override frequencyPenalty on retry after stream error', async () => {
        // Arrange
        const baseModel = new MockLanguageModel({
          doStream: {
            stream: convertArrayToReadableStream([
              { type: 'stream-start', warnings: [] },
              { type: 'error', error: new Error('Stream error') },
            ]),
          },
        });

        const fallbackModel = new MockLanguageModel({
          doStream: {
            stream: convertArrayToReadableStream(mockStreamChunks),
          },
        });

        const retryableModel = createRetryable({
          model: baseModel,
          retries: [
            { model: fallbackModel, options: { frequencyPenalty: 0.8 } },
          ],
        });

        // Act
        const result = streamText({
          model: retryableModel,
          prompt: 'Hello!',
          frequencyPenalty: 0.2,
        });

        await convertAsyncIterableToArray(result.fullStream);

        // Assert
        expect(baseModel.doStream).toHaveBeenCalledWith(
          expect.objectContaining({ frequencyPenalty: 0.2 }),
        );
        expect(fallbackModel.doStream).toHaveBeenCalledWith(
          expect.objectContaining({ frequencyPenalty: 0.8 }),
        );
      });
    });

    describe('headers', () => {
      it('should override headers on retry after stream error', async () => {
        // Arrange
        const baseModel = new MockLanguageModel({
          doStream: {
            stream: convertArrayToReadableStream([
              { type: 'stream-start', warnings: [] },
              { type: 'error', error: new Error('Stream error') },
            ]),
          },
        });

        const fallbackModel = new MockLanguageModel({
          doStream: {
            stream: convertArrayToReadableStream(mockStreamChunks),
          },
        });

        const retryHeaders = { 'x-retry': 'retry' };

        const retryableModel = createRetryable({
          model: baseModel,
          retries: [
            { model: fallbackModel, options: { headers: retryHeaders } },
          ],
        });

        // Act
        const result = streamText({
          model: retryableModel,
          prompt: 'Hello!',
        });

        await convertAsyncIterableToArray(result.fullStream);

        // Assert
        expect(fallbackModel.doStream).toHaveBeenCalledWith(
          expect.objectContaining({
            headers: expect.objectContaining({
              'x-retry': 'retry',
            }),
          }),
        );
      });
    });
  });

  describe('RetryError', () => {
    it('should throw RetryError when all retry attempts are exhausted', async () => {
      // Arrange
      const baseModel = new MockLanguageModel({ doStream: retryableError });
      const fallbackModel1 = new MockLanguageModel({
        doStream: nonRetryableError,
      });
      const fallbackModel2 = new MockLanguageModel({
        doStream: retryableError,
      });

      const retryableModel = createRetryable({
        model: baseModel,
        retries: [fallbackModel1, fallbackModel2],
      });

      const result = streamText({
        model: retryableModel,
        prompt,
      });

      // Act
      const chunks = await convertAsyncIterableToArray(result.fullStream);

      // Assert
      expect(baseModel.doStream).toHaveBeenCalledTimes(1);
      expect(fallbackModel1.doStream).toHaveBeenCalledTimes(1);
      expect(fallbackModel2.doStream).toHaveBeenCalledTimes(1);
      expect(chunks).toMatchInlineSnapshot(`
        [
          {
            "type": "start",
          },
          {
            "error": [AI_RetryError: Failed after 3 attempts. Last error: Rate limit exceeded],
            "type": "error",
          },
        ]
      `);

      const errorChunk = errorFromChunks(chunks);
      expect(errorChunk).toBeDefined();
      expect(errorChunk).toBeInstanceOf(RetryError);

      const retryError = errorChunk as RetryError;
      expect(retryError.reason).toBe('maxRetriesExceeded');
      expect(retryError.errors).toHaveLength(3);
      expect(retryError.errors[0]).toBe(retryableError);
      expect(retryError.errors[1]).toBe(nonRetryableError);
      expect(retryError.errors[2]).toBe(retryableError);
    });

    it('should throw original error directly on first attempt with no retryables', async () => {
      // Arrange
      const baseModel = new MockLanguageModel({ doStream: retryableError });

      const retryableModel = createRetryable({
        model: baseModel,
        retries: [], // No retry models
      });

      const result = streamText({
        model: retryableModel,
        prompt,
        maxRetries: 0, // No automatic retries
      });

      // Act
      const chunks = await convertAsyncIterableToArray(result.fullStream);

      // Assert
      expect(baseModel.doStream).toHaveBeenCalledTimes(1);
      expect(errorFromChunks(chunks)).toBe(retryableError);
      expect(chunks).toMatchInlineSnapshot(`
        [
          {
            "type": "start",
          },
          {
            "error": [AI_APICallError: Rate limit exceeded],
            "type": "error",
          },
        ]
      `);
    });

    it('should throw original error directly when retryable returns undefined', async () => {
      // Arrange
      const baseModel = new MockLanguageModel({ doStream: nonRetryableError });

      const retryableModel = createRetryable({
        model: baseModel,
        retries: [() => undefined],
      });

      const result = streamText({
        model: retryableModel,
        prompt,
        maxRetries: 0, // No automatic retries
      });

      // Act
      const chunks = await convertAsyncIterableToArray(result.fullStream);

      // Assert
      expect(baseModel.doStream).toHaveBeenCalledTimes(1);
      expect(errorFromChunks(chunks)).toBe(nonRetryableError);
      expect(chunks).toMatchInlineSnapshot(`
        [
          {
            "type": "start",
          },
          {
            "error": [AI_APICallError: Invalid API key],
            "type": "error",
          },
        ]
      `);
    });
  });
});
