import type { LanguageModelV2StreamPart } from '@ai-sdk/provider';
import {
  convertArrayToReadableStream,
  convertAsyncIterableToArray,
} from '@ai-sdk/provider-utils/test';
import { APICallError, generateText, RetryError, streamText } from 'ai';
import { describe, expect, it, vi } from 'vitest';
import {
  type CreateRetryableOptions,
  createRetryable,
  isErrorAttempt,
  isResultAttempt,
  type Retryable,
} from './create-retryable-model.js';
import {
  chunksToText,
  createMockModel,
  createMockStreamingModel,
  errorFromChunks,
} from './test-utils.js';
import type { LanguageModelV2Generate } from './types.js';

type OnError = Required<CreateRetryableOptions>['onError'];
type OnRetry = Required<CreateRetryableOptions>['onRetry'];

const prompt = 'Hello!';

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

const testUsage = {
  inputTokens: 3,
  outputTokens: 10,
  totalTokens: 13,
  reasoningTokens: undefined,
  cachedInputTokens: undefined,
};

const mockStreamChunks: LanguageModelV2StreamPart[] = [
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

      const fallbackRetryable: Retryable = (context) => {
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

    it('should use plain language models for error-based attempts', async () => {
      // Arrange
      const baseModel = createMockModel(retryableError);
      const fallbackModel1 = createMockModel(mockResult);
      const fallbackModel2 = createMockModel(mockResult);

      const fallbackRetryable: Retryable = (context) => {
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
      expect(baseModel.doGenerateCalls.length).toBe(1);
      expect(fallbackModel1.doGenerateCalls.length).toBe(1); // Should be called
      expect(fallbackModel2.doGenerateCalls.length).toBe(0);
      expect(result.text).toBe(mockResultText);
    });
  });

  describe('result-based retries', () => {
    it('should retry with results', async () => {
      // Arrange
      const baseModel = createMockModel(contentFilterResult);
      const fallbackModel = createMockModel(mockResult);
      const fallbackRetryable: Retryable = (context) => {
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
      expect(baseModel.doGenerateCalls.length).toBe(1);
      expect(fallbackModel.doGenerateCalls.length).toBe(1);
      expect(result.text).toBe(mockResultText);
    });

    it('should ignore plain language models for result-based attempts', async () => {
      // Arrange
      const baseModel = createMockModel(contentFilterResult);
      const fallbackModel1 = createMockModel(mockResult);
      const fallbackModel2 = createMockModel(mockResult);

      const fallbackRetryable: Retryable = (context) => {
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
          retries: [fallbackModel1, fallbackRetryable],
        }),
        prompt: 'Hello!',
      });

      // Assert
      expect(baseModel.doGenerateCalls.length).toBe(1);
      expect(fallbackModel1.doGenerateCalls.length).toBe(0); // Should not be called
      expect(fallbackModel2.doGenerateCalls.length).toBe(1);
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
});

describe('streamText', () => {
  it('should stream successfully when no errors occur', async () => {
    const baseModel = createMockStreamingModel({
      stream: convertArrayToReadableStream(mockStreamChunks),
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

    expect(baseModel.doStreamCalls.length).toBe(1);
    expect(chunksToText(chunks)).toBe('Hello, world!');
  });

  it('should retry when error occurs before stream starts', async () => {
    const baseModel = createMockStreamingModel(retryableError);

    const fallbackModel = createMockStreamingModel({
      stream: convertArrayToReadableStream(mockStreamChunks),
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

    expect(baseModel.doStreamCalls.length).toBe(1);
    expect(fallbackModel.doStreamCalls.length).toBe(1);
    expect(chunksToText(chunks)).toBe('Hello, world!');
  });

  // TODO: implement retrying errors at the beginning of the stream
  it.skip('should retry when error occurs at the stream start', async () => {
    const baseModel = createMockStreamingModel({
      stream: convertArrayToReadableStream([
        { type: 'stream-start', warnings: [] },
        {
          type: 'error',
          error: { type: 'overloaded_error', message: 'Overloaded' },
        },
      ]),
    });

    const fallbackModel = createMockStreamingModel({
      stream: convertArrayToReadableStream(mockStreamChunks),
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

    expect(baseModel.doStreamCalls.length).toBe(1);
    expect(fallbackModel.doStreamCalls.length).toBe(1);
    expect(chunks).toMatchInlineSnapshot();
  });

  it.skip('should not retry when error occurs during mid-stream', async () => {
    const baseModel = createMockStreamingModel({
      stream: convertArrayToReadableStream([
        { type: 'stream-start', warnings: [] },
        { type: 'text-start', id: '1' },
        { type: 'text-delta', id: '1', delta: 'Hello' },
        { type: 'error', error: new Error('Overloaded') },
      ]),
    });

    const fallbackModel = createMockStreamingModel({
      stream: convertArrayToReadableStream(mockStreamChunks),
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

    expect(baseModel.doStreamCalls.length).toBe(1);
    expect(fallbackModel.doStreamCalls.length).toBe(0);
    expect(chunks).toMatchInlineSnapshot();
  });

  describe('onError', () => {
    it('should call onError handler when an error occurs', async () => {
      // Arrange
      const baseModel = createMockStreamingModel(retryableError);
      const fallbackModel = createMockStreamingModel({
        stream: convertArrayToReadableStream(mockStreamChunks),
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
      expect(firstErrorCall.totalAttempts).toBe(1);
    });

    it('should call onError handler for each error in multiple attempts', async () => {
      // Arrange
      const baseModel = createMockStreamingModel(retryableError);
      const fallbackModel = createMockStreamingModel(nonRetryableError);
      const finalModel = createMockStreamingModel({
        stream: convertArrayToReadableStream(mockStreamChunks),
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
      expect(firstErrorCall.totalAttempts).toBe(1);

      expect(secondErrorCall.current.error).toBe(nonRetryableError);
      expect(secondErrorCall.current.model).toBe(fallbackModel);
      expect(secondErrorCall.totalAttempts).toBe(2);
    });

    it('should not call onError handler when streaming succeeds', async () => {
      // Arrange
      const baseModel = createMockStreamingModel({
        stream: convertArrayToReadableStream(mockStreamChunks),
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
      const baseModel = createMockStreamingModel(retryableError);
      const fallbackModel = createMockStreamingModel({
        stream: convertArrayToReadableStream(mockStreamChunks),
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
      expect(firstRetryCall.totalAttempts).toBe(1);
    });

    it('should call onRetry handler for each retry attempt', async () => {
      // Arrange
      const baseModel = createMockStreamingModel(retryableError);
      const fallbackModel1 = createMockStreamingModel(nonRetryableError);
      const fallbackModel2 = createMockStreamingModel({
        stream: convertArrayToReadableStream(mockStreamChunks),
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
      const baseModel = createMockStreamingModel({
        stream: convertArrayToReadableStream(mockStreamChunks),
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

  describe('maxAttempts', () => {
    it('should try each model only once by default', async () => {
      // Arrange
      const baseModel = createMockStreamingModel(retryableError);
      const fallbackModel1 = createMockStreamingModel(retryableError);
      const fallbackModel2 = createMockStreamingModel(retryableError);
      const finalModel = createMockStreamingModel({
        stream: convertArrayToReadableStream(mockStreamChunks),
      });

      // Act
      const retryableModel = createRetryable({
        model: baseModel,
        retries: [
          fallbackModel1,
          () => ({ model: fallbackModel2 }),
          async () => ({ model: finalModel }),
        ],
      });

      const result = streamText({
        model: retryableModel,
        prompt,
      });

      await convertAsyncIterableToArray(result.fullStream);

      // Assert
      expect(baseModel.doStreamCalls.length).toBe(1);
      expect(fallbackModel1.doStreamCalls.length).toBe(1);
      expect(fallbackModel2.doStreamCalls.length).toBe(1);
      expect(finalModel.doStreamCalls.length).toBe(1);
    });

    it('should try models multiple times if maxAttempts is set', async () => {
      // Arrange
      const baseModel = createMockStreamingModel(retryableError);
      const fallbackModel1 = createMockStreamingModel(retryableError);
      const fallbackModel2 = createMockStreamingModel(retryableError);
      const finalModel = createMockStreamingModel({
        stream: convertArrayToReadableStream(mockStreamChunks),
      });

      // Act
      const retryableModel = createRetryable({
        model: baseModel,
        retries: [
          // Retryable with different maxAttempts
          () => ({ model: fallbackModel1, maxAttempts: 2 }),
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
      expect(baseModel.doStreamCalls.length).toBe(1);
      expect(fallbackModel1.doStreamCalls.length).toBe(2);
      expect(fallbackModel2.doStreamCalls.length).toBe(3);
      expect(finalModel.doStreamCalls.length).toBe(1);
    });
  });

  describe('RetryError', () => {
    it('should throw RetryError when all retry attempts are exhausted', async () => {
      // Arrange
      const baseModel = createMockStreamingModel(retryableError);
      const fallbackModel1 = createMockStreamingModel(nonRetryableError);
      const fallbackModel2 = createMockStreamingModel(retryableError);

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
      expect(baseModel.doStreamCalls.length).toBe(1);
      expect(fallbackModel1.doStreamCalls.length).toBe(1);
      expect(fallbackModel2.doStreamCalls.length).toBe(1);
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
      const baseModel = createMockStreamingModel(retryableError);

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
      expect(baseModel.doStreamCalls.length).toBe(1);
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
      const baseModel = createMockStreamingModel(nonRetryableError);

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
      expect(baseModel.doStreamCalls.length).toBe(1);
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
