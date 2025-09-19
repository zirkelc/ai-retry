import type { LanguageModelV2 } from '@ai-sdk/provider';
import { APICallError, generateText } from 'ai';
import { expect, it } from 'vitest';
import { createRetryable } from './create-retryable-model.js';
import { createMockModel } from './test-utils.js';

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

it('should not fallback without errors', async () => {
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

it('should retry with fallback models', async () => {
  // Arrange
  const baseModel = createMockModel(retryableError);
  const fallbackModel1 = createMockModel(nonRetryableError);
  const fallbackModel2 = createMockModel(mockResult);

  // Act
  const result = await generateText({
    model: createRetryable({
      model: baseModel,
      retries: [fallbackModel1, fallbackModel2],
    }),
    prompt: 'Hello!',
    maxRetries: 0,
  });

  // Assert
  expect(baseModel.doGenerateCalls.length).toBe(1);
  expect(fallbackModel1.doGenerateCalls.length).toBe(1);
  expect(fallbackModel2.doGenerateCalls.length).toBe(1);
  expect(result.text).toBe(mockResultText);
});
