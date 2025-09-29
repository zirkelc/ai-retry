import { anthropic } from '@ai-sdk/anthropic';
import type { LanguageModelV2StreamPart } from '@ai-sdk/provider';
import {
  convertArrayToReadableStream,
  convertAsyncIterableToArray,
} from '@ai-sdk/provider-utils/test';
import { APICallError, generateText, streamText } from 'ai';
import { describe, expect, it, vi } from 'vitest';
import { createRetryable } from '../create-retryable-model.js';
import { createMockModel, createMockStreamingModel } from '../test-utils.js';
import type { LanguageModelV2Generate } from '../types.js';
import {
  type AnthropicErrorResponse,
  anthropicServiceOverloaded,
} from './anthropic-service-overloaded.js';

const mockResultText = 'Hello, world!';

const mockResult: LanguageModelV2Generate = {
  finishReason: 'stop',
  usage: { inputTokens: 10, outputTokens: 20, totalTokens: 30 },
  content: [{ type: 'text', text: mockResultText }],
  warnings: [],
};

const overloadedError: AnthropicErrorResponse = {
  type: 'error',
  error: { type: 'overloaded_error', message: 'Overloaded' },
};

process.env.ANTHROPIC_API_KEY = 'test';

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
  { type: 'text-delta', id: '1', delta: 'world!' },
  { type: 'text-end', id: '1' },
  {
    type: 'finish',
    finishReason: 'stop',
    usage: { inputTokens: 10, outputTokens: 20, totalTokens: 30 },
  },
];

// TODO change to always use anthropic() with a mocked fetch response
describe('anthropicServiceOverloaded', () => {
  describe('generateText', () => {
    it('should succeed without errors', async () => {
      // Arrange
      const baseModel = createMockModel(mockResult);
      const retryModel = createMockModel(mockResult);

      // Act
      const result = await generateText({
        model: createRetryable({
          model: baseModel,
          retries: [anthropicServiceOverloaded(retryModel)],
        }),
        prompt: 'Hello!',
      });

      // Assert
      expect(baseModel.doGenerateCalls.length).toBe(1);
      expect(result.text).toBe(mockResultText);
    });

    it('should retry for status 200 with overloaded error', async () => {
      // Arrange
      const baseModel = anthropic('claude-sonnet-4-20250514');
      const retryModel = createMockModel(mockResult);

      vi.spyOn(globalThis, 'fetch').mockResolvedValueOnce(
        new Response(JSON.stringify(overloadedError), {
          status: 200, // Anthropic sometimes returns 200 even for errors
          headers: { 'Content-Type': 'application/json' },
        }),
      );

      const baseModelSpy = vi.spyOn(baseModel, 'doGenerate');

      // Act
      const result = await generateText({
        model: createRetryable({
          model: baseModel,
          retries: [anthropicServiceOverloaded(retryModel)],
        }),
        prompt: 'Hello!',
      });

      // Assert
      expect(baseModelSpy).toHaveBeenCalledTimes(1);
      expect(retryModel.doGenerateCalls.length).toBe(1);
      expect(result.text).toBe(mockResultText);
    });

    it('should retry for status 529 with overloaded error', async () => {
      // Arrange
      const baseModel = anthropic('claude-sonnet-4-20250514');
      const retryModel = createMockModel(mockResult);

      vi.spyOn(globalThis, 'fetch').mockResolvedValueOnce(
        new Response(JSON.stringify(overloadedError), {
          status: 529,
          headers: { 'Content-Type': 'application/json' },
        }),
      );

      const baseModelSpy = vi.spyOn(baseModel, 'doGenerate');

      // Act
      const result = await generateText({
        model: createRetryable({
          model: baseModel,
          retries: [anthropicServiceOverloaded(retryModel)],
        }),
        prompt: 'Hello!',
      });

      // Assert
      expect(baseModelSpy).toHaveBeenCalledTimes(1);
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
          retries: [anthropicServiceOverloaded(retryModel)],
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
});
