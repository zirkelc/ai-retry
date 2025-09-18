import type { LanguageModelV2 } from '@ai-sdk/provider';
import { APICallError, generateText } from 'ai';
import { describe, expect, it } from 'vitest';
import { createRetryable } from '../create-retryable-model.js';
import { createMockModel } from '../test-utils.js';
import { fallbackAfterTimeout } from './fallback-after-timeout.js';

type LanguageModelV2GenerateFn = LanguageModelV2['doGenerate'];

type LanguageModelV2GenerateResult = Awaited<
  ReturnType<LanguageModelV2GenerateFn>
>;

const mockResultText = 'Hello, world!';

const mockResult: LanguageModelV2GenerateResult = {
  finishReason: 'stop',
  usage: { inputTokens: 10, outputTokens: 20, totalTokens: 30 },
  content: [{ type: 'text', text: mockResultText }],
  warnings: [],
};

const genericError = new APICallError({
  message: 'Some other error',
  url: '',
  requestBodyValues: {},
  statusCode: 500,
  responseHeaders: {},
  responseBody: '{"error": {"message": "Internal server error"}}',
  isRetryable: true,
  data: {
    error: {
      message: 'Internal server error',
    },
  },
});

describe('fallbackAfterTimeout', () => {
  it('should succeed without errors', async () => {
    // Arrange
    const baseModel = createMockModel(mockResult);
    const retryModel = createMockModel(mockResult);

    // Act
    const result = await generateText({
      model: createRetryable({
        model: baseModel,
        retries: [fallbackAfterTimeout(retryModel)],
      }),
      prompt: 'Hello!',
    });

    // Assert
    expect(baseModel.doGenerateCalls.length).toBe(1);
    expect(result.text).toBe(mockResultText);
  });

  it('should fallback in case of timeout error', async () => {
    // Arrange
    const baseModel = createMockModel(async (opts) => {
      // Check if abortSignal is aborted and throw appropriate error
      if (opts.abortSignal?.aborted) {
        throw new DOMException('The operation was aborted', 'AbortError');
      }

      // Listen for abort event during the async operation
      return new Promise((resolve, reject) => {
        const timeout = setTimeout(() => {
          resolve(mockResult);
        }, 1_000);

        opts.abortSignal?.addEventListener('abort', () => {
          clearTimeout(timeout);
          reject(new DOMException('The operation was aborted', 'AbortError'));
        });
      });
    });
    const retryModel = createMockModel(mockResult);

    // Act
    const result = await generateText({
      model: createRetryable({
        model: baseModel,
        retries: [fallbackAfterTimeout(retryModel)],
      }),
      prompt: 'Hello!',
      maxRetries: 0,
      abortSignal: AbortSignal.timeout(100), // Very short timeout to trigger timeout
    });

    // Assert
    expect(baseModel.doGenerateCalls.length).toBe(1);
    expect(retryModel.doGenerateCalls.length).toBe(1);
    expect(result.text).toBe(mockResultText);
  });

  it('should not fallback for non-timeout errors', async () => {
    // Arrange
    const baseModel = createMockModel(genericError);
    const retryModel = createMockModel(mockResult);

    // Act
    const result = generateText({
      model: createRetryable({
        model: baseModel,
        retries: [fallbackAfterTimeout(retryModel)],
      }),
      prompt: 'Hello!',
      maxRetries: 0,
    });

    // Assert
    await expect(result).rejects.toThrowError(APICallError);
    expect(baseModel.doGenerateCalls.length).toBe(1);
    expect(retryModel.doGenerateCalls.length).toBe(0);
  });
});
