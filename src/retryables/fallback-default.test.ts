import type { LanguageModelV2 } from '@ai-sdk/provider';
import { APICallError, generateText } from 'ai';
import { describe, expect, it } from 'vitest';
import { createRetryable } from '../create-retryable-model.js';
import { createMockModel } from '../test-utils.js';

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

const apiCallError = () =>
  new APICallError({
    message: 'Rate limit exceeded',
    url: '',
    requestBodyValues: {},
    statusCode: 429,
    responseHeaders: {},
    responseBody: '{"error": {"message": "Rate limit exceeded"}}',
    isRetryable: false,
  });

describe('fallback default behavior', () => {
  it('should succeed without errors', async () => {
    // Arrange
    const baseModel = createMockModel(mockResult);
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
    expect(result.text).toBe(mockResultText);
  });

  it('should retry with fallback models', async () => {
    // Arrange
    const baseModel = createMockModel(apiCallError());

    const fallbackModel1 = createMockModel(apiCallError());

    const fallbackModel2 = createMockModel(mockResult);

    const retryableModel = createRetryable({
      model: baseModel,
      retries: [fallbackModel1, fallbackModel2],
    });

    // Act
    const result = await generateText({
      model: retryableModel,
      prompt: 'Hello!',
      maxRetries: 0,
    });

    // Assert
    expect(baseModel.doGenerateCalls.length).toBe(1);
    expect(fallbackModel1.doGenerateCalls.length).toBe(1);
    expect(fallbackModel2.doGenerateCalls.length).toBe(1);
    expect(result.text).toBe(mockResultText);
  });
});
