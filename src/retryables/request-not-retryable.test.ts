import type { LanguageModelV2 } from '@ai-sdk/provider';
import { APICallError, generateText } from 'ai';
import { describe, expect, it } from 'vitest';
import { createRetryable } from '../create-retryable-model.js';
import { createMockModel } from '../test-utils.js';
import { requestNotRetryable } from './request-not-retryable.js';

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

describe('requestNotRetryable', () => {
  it('should succeed without errors', async () => {
    // Arrange
    const baseModel = createMockModel(mockResult);
    const retryModel = createMockModel(mockResult);

    // Act
    const result = await generateText({
      model: createRetryable({
        model: baseModel,
        retries: [requestNotRetryable(retryModel)],
      }),
      prompt: 'Hello!',
    });

    // Assert
    expect(baseModel.doGenerateCalls.length).toBe(1);
    expect(result.text).toBe(mockResultText);
  });

  it('should fallback in case of non-retryable error', async () => {
    // Arrange
    const baseModel = createMockModel(nonRetryableError);
    const retryModel = createMockModel(mockResult);

    // Act
    const result = await generateText({
      model: createRetryable({
        model: baseModel,
        retries: [requestNotRetryable(retryModel)],
      }),
      prompt: 'Hello!',
      maxRetries: 0,
    });

    // Assert
    expect(baseModel.doGenerateCalls.length).toBe(1);
    expect(retryModel.doGenerateCalls.length).toBe(1);
    expect(result.text).toBe(mockResultText);
  });

  it('should not fallback for retryable error', async () => {
    // Arrange
    const baseModel = createMockModel(retryableError);
    const retryModel = createMockModel(mockResult);

    // Act
    const result = generateText({
      model: createRetryable({
        model: baseModel,
        retries: [requestNotRetryable(retryModel)],
      }),
      prompt: 'Hello!',
      maxRetries: 0,
    });

    // Assert
    await expect(result).rejects.toThrowError(APICallError);
    expect(baseModel.doGenerateCalls.length).toBe(1);
    expect(retryModel.doGenerateCalls.length).toBe(0);
  });

  it('should not fallback for non AI SDK errors', async () => {
    // Arrange
    const genericError = new Error('Some generic error');
    const baseModel = createMockModel(genericError);
    const retryModel = createMockModel(mockResult);

    // Act
    const result = generateText({
      model: createRetryable({
        model: baseModel,
        retries: [requestNotRetryable(retryModel)],
      }),
      prompt: 'Hello!',
      maxRetries: 0,
    });

    // Assert
    await expect(result).rejects.toThrowError(Error);
    expect(baseModel.doGenerateCalls.length).toBe(1);
    expect(retryModel.doGenerateCalls.length).toBe(0);
  });
});
