import {
  APICallError,
  generateText,
  NoObjectGeneratedError,
  TypeValidationError,
} from 'ai';
import { describe, expect, it } from 'vitest';
import z from 'zod';
import { createRetryable } from '../create-retryable-model.js';
import { createMockModel } from '../test-utils.js';
import type { LanguageModelV2Generate } from '../types.js';
import { responseSchemaMismatch } from './response-schema-mismatch.js';

const mockResultText = 'Hello, world!';

const mockResult: LanguageModelV2Generate = {
  finishReason: 'stop',
  usage: { inputTokens: 10, outputTokens: 20, totalTokens: 30 },
  content: [{ type: 'text', text: mockResultText }],
  warnings: [],
};

const schema = z.object({
  values: z.array(z.string()),
});
const value = [{ invalid: 'data' }];
const zodError = schema.safeParse(value).error;

const error = new NoObjectGeneratedError({
  response: {
    id: 'mock-response-id',
    modelId: 'mock-model-id',
    timestamp: new Date(),
  },
  finishReason: 'stop',
  cause: new TypeValidationError({
    value,
    cause: zodError,
  }),
  usage: { inputTokens: 10, outputTokens: 20, totalTokens: 30 },
});

describe('responseSchemaMismatch', () => {
  it('should succeed without errors', async () => {
    // Arrange
    const baseModel = createMockModel(mockResult);
    const retryModel = createMockModel(mockResult);

    // Act
    const result = await generateText({
      model: createRetryable({
        model: baseModel,
        retries: [responseSchemaMismatch(retryModel)],
      }),
      prompt: 'Hello!',
    });

    // Assert
    expect(baseModel.doGenerateCalls.length).toBe(1);
    expect(result.text).toBe(mockResultText);
  });

  it('should retry in case of error', async () => {
    // Arrange
    const baseModel = createMockModel(error);
    const retryModel = createMockModel(mockResult);

    // Act
    const result = await generateText({
      model: createRetryable({
        model: baseModel,
        retries: [responseSchemaMismatch(retryModel)],
      }),
      prompt: 'Hello!',
      maxRetries: 0,
    });

    // Assert
    expect(baseModel.doGenerateCalls.length).toBe(1);
    expect(retryModel.doGenerateCalls.length).toBe(1);
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
        retries: [responseSchemaMismatch(retryModel)],
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
