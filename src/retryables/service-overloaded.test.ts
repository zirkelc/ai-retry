import { APICallError, generateText } from 'ai';
import { describe, expect, it } from 'vitest';
import { createRetryable } from '../create-retryable-model.js';
import { createMockModel } from '../test-utils.js';
import type { LanguageModelV2Generate } from '../types.js';
import { serviceOverloaded } from './service-overloaded.js';

const mockResultText = 'Hello, world!';

const mockResult: LanguageModelV2Generate = {
  finishReason: 'stop',
  usage: { inputTokens: 10, outputTokens: 20, totalTokens: 30 },
  content: [{ type: 'text', text: mockResultText }],
  warnings: [],
};

const overloadedError = new APICallError({
  message:
    'Overloaded: The server is currently overloaded. Please try again later.',
  url: '',
  requestBodyValues: {},
  statusCode: 529,
  responseHeaders: {},
  responseBody: '',
  isRetryable: false,
  data: {},
});

describe('serviceOverloaded', () => {
  it('should succeed without errors', async () => {
    // Arrange
    const baseModel = createMockModel(mockResult);
    const retryModel = createMockModel(mockResult);

    // Act
    const result = await generateText({
      model: createRetryable({
        model: baseModel,
        retries: [serviceOverloaded(retryModel)],
      }),
      prompt: 'Hello!',
    });

    // Assert
    expect(baseModel.doGenerateCalls.length).toBe(1);
    expect(result.text).toBe(mockResultText);
  });

  it('should retry for status 529', async () => {
    // Arrange
    const baseModel = createMockModel(overloadedError);
    const retryModel = createMockModel(mockResult);

    // Act
    const result = await generateText({
      model: createRetryable({
        model: baseModel,
        retries: [serviceOverloaded(retryModel)],
      }),
      prompt: 'Hello!',
    });

    // Assert
    expect(baseModel.doGenerateCalls.length).toBe(1);
    expect(retryModel.doGenerateCalls.length).toBe(1);
    expect(result.text).toBe(mockResultText);
  });

  it('should not retry for status 200', async () => {
    // Arrange
    const baseModel = createMockModel(mockResult);
    const retryModel = createMockModel(mockResult);

    // Act
    const result = await generateText({
      model: createRetryable({
        model: baseModel,
        retries: [serviceOverloaded(retryModel)],
      }),
      prompt: 'Hello!',
    });

    // Assert
    expect(baseModel.doGenerateCalls.length).toBe(1);
    expect(retryModel.doGenerateCalls.length).toBe(0);
    expect(result.text).toBe(mockResultText);
  });

  it('should not retry if no matches', async () => {
    // Arrange
    const baseModel = createMockModel(
      new APICallError({
        message: 'Some other error',
        url: '',
        requestBodyValues: {},
        statusCode: 400,
        responseHeaders: {},
        responseBody: '{}',
        isRetryable: false,
        data: {
          error: {
            message: 'Some other error',
            type: null,
            param: 'prompt',
            code: 'other_error',
          },
        },
      }),
    );

    const retryModel = createMockModel(mockResult);

    // Act
    const result = generateText({
      model: createRetryable({
        model: baseModel,
        retries: [serviceOverloaded(retryModel)],
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
