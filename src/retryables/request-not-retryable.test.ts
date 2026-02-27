import {
  convertArrayToReadableStream,
  convertAsyncIterableToArray,
} from '@ai-sdk/provider-utils/test';
import {
  APICallError,
  embed,
  generateImage,
  generateText,
  streamText,
} from 'ai';
import { describe, expect, it } from 'vitest';
import { createRetryable } from '../create-retryable-model.js';
import {
  chunksToText,
  MockEmbeddingModel,
  MockImageModel,
  MockLanguageModel,
} from '../test-utils.js';
import type {
  EmbeddingModelEmbed,
  ImageModelGenerate,
  LanguageModelGenerate,
  LanguageModelStreamPart,
} from '../types.js';
import { requestNotRetryable } from './request-not-retryable.js';

const mockResultText = 'Hello, world!';

const mockResult: LanguageModelGenerate = {
  finishReason: { unified: 'stop', raw: undefined },
  usage: {
    inputTokens: { total: 10, noCache: 0, cacheRead: 0, cacheWrite: 0 },
    outputTokens: { total: 20, text: 0, reasoning: 0 },
  },
  content: [{ type: 'text', text: mockResultText }],
  warnings: [],
};

const mockEmbeddings: EmbeddingModelEmbed = {
  embeddings: [[0.1, 0.2, 0.3]],
  usage: { tokens: 5 },
  warnings: [],
};

/** Valid base64 PNG image (1x1 transparent pixel) */
const validBase64Image = `iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg==`;

const mockImageResult: ImageModelGenerate = {
  images: [validBase64Image],
  warnings: [],
  response: {
    timestamp: new Date(),
    modelId: `mock-model`,
    headers: undefined,
  },
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
  { type: 'text-delta', id: '1', delta: 'world!' },
  { type: 'text-end', id: '1' },
  {
    type: 'finish',
    finishReason: { unified: 'stop', raw: undefined },
    usage: {
      inputTokens: { total: 10, noCache: 0, cacheRead: 0, cacheWrite: 0 },
      outputTokens: { total: 20, text: 0, reasoning: 0 },
    },
  },
];

describe('requestNotRetryable', () => {
  describe('generateText', () => {
    it('should succeed without errors', async () => {
      // Arrange
      const baseModel = new MockLanguageModel({ doGenerate: mockResult });
      const retryModel = new MockLanguageModel({ doGenerate: mockResult });

      // Act
      const result = await generateText({
        model: createRetryable({
          model: baseModel,
          retries: [requestNotRetryable(retryModel)],
        }),
        prompt: 'Hello!',
      });

      // Assert
      expect(baseModel.doGenerate).toHaveBeenCalledTimes(1);
      expect(result.text).toBe(mockResultText);
    });

    it('should fallback in case of non-retryable error', async () => {
      // Arrange
      const baseModel = new MockLanguageModel({
        doGenerate: nonRetryableError,
      });
      const retryModel = new MockLanguageModel({ doGenerate: mockResult });

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
      expect(baseModel.doGenerate).toHaveBeenCalledTimes(1);
      expect(retryModel.doGenerate).toHaveBeenCalledTimes(1);
      expect(result.text).toBe(mockResultText);
    });

    it('should not fallback for retryable error', async () => {
      // Arrange
      const baseModel = new MockLanguageModel({ doGenerate: retryableError });
      const retryModel = new MockLanguageModel({ doGenerate: mockResult });

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
      expect(baseModel.doGenerate).toHaveBeenCalledTimes(1);
      expect(retryModel.doGenerate).toHaveBeenCalledTimes(0);
    });

    it('should not fallback for non AI SDK errors', async () => {
      // Arrange
      const genericError = new Error('Some generic error');
      const baseModel = new MockLanguageModel({ doGenerate: genericError });
      const retryModel = new MockLanguageModel({ doGenerate: mockResult });

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
      expect(baseModel.doGenerate).toHaveBeenCalledTimes(1);
      expect(retryModel.doGenerate).toHaveBeenCalledTimes(0);
    });
  });

  describe('streamText', () => {
    it('should succeed without errors', async () => {
      // Arrange
      const baseModel = new MockLanguageModel({
        doStream: {
          stream: convertArrayToReadableStream(mockStreamChunks),
        },
      });
      const retryModel = new MockLanguageModel({
        doStream: {
          stream: convertArrayToReadableStream(mockStreamChunks),
        },
      });
      let error: unknown;

      // Act
      const result = streamText({
        model: createRetryable({
          model: baseModel,
          retries: [requestNotRetryable(retryModel)],
        }),
        prompt: 'Hello!',
        onError(data) {
          error = data.error;
        },
      });

      const chunks = await convertAsyncIterableToArray(result.fullStream);

      // Assert
      expect(baseModel.doStream).toHaveBeenCalledTimes(1);
      expect(retryModel.doStream).toHaveBeenCalledTimes(0);
      expect(error).toBeUndefined();
      expect(chunksToText(chunks)).toBe(mockResultText);
    });

    it('should fallback in case of non-retryable error', async () => {
      // Arrange
      const baseModel = new MockLanguageModel({ doStream: nonRetryableError });
      const retryModel = new MockLanguageModel({
        doStream: {
          stream: convertArrayToReadableStream(mockStreamChunks),
        },
      });
      let error: unknown;

      // Act
      const result = streamText({
        model: createRetryable({
          model: baseModel,
          retries: [requestNotRetryable(retryModel)],
        }),
        prompt: 'Hello!',
        maxRetries: 0,
        onError(data) {
          error = data.error;
        },
      });

      const chunks = await convertAsyncIterableToArray(result.fullStream);

      // Assert
      expect(baseModel.doStream).toHaveBeenCalledTimes(1);
      expect(retryModel.doStream).toHaveBeenCalledTimes(1);
      expect(error).toBeUndefined();
      expect(chunksToText(chunks)).toBe(mockResultText);
    });

    it('should not fallback for retryable error', async () => {
      // Arrange
      const baseModel = new MockLanguageModel({ doStream: retryableError });
      const retryModel = new MockLanguageModel({
        doStream: {
          stream: convertArrayToReadableStream(mockStreamChunks),
        },
      });
      let error: unknown;

      // Act
      const result = streamText({
        model: createRetryable({
          model: baseModel,
          retries: [requestNotRetryable(retryModel)],
        }),
        prompt: 'Hello!',
        maxRetries: 0,
        onError(data) {
          error = data.error;
        },
      });

      const chunks = await convertAsyncIterableToArray(result.fullStream);

      // Assert
      expect(baseModel.doStream).toHaveBeenCalledTimes(1);
      expect(retryModel.doStream).toHaveBeenCalledTimes(0);
      expect(error).toBeDefined();
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

    it('should not fallback for non AI SDK errors', async () => {
      // Arrange
      const genericError = new Error('Some generic error');
      const baseModel = new MockLanguageModel({ doStream: genericError });
      const retryModel = new MockLanguageModel({
        doStream: {
          stream: convertArrayToReadableStream(mockStreamChunks),
        },
      });
      let error: unknown;

      // Act
      const result = streamText({
        model: createRetryable({
          model: baseModel,
          retries: [requestNotRetryable(retryModel)],
        }),
        prompt: 'Hello!',
        maxRetries: 0,
        onError(data) {
          error = data.error;
        },
      });

      const chunks = await convertAsyncIterableToArray(result.fullStream);

      // Assert
      expect(baseModel.doStream).toHaveBeenCalledTimes(1);
      expect(retryModel.doStream).toHaveBeenCalledTimes(0);
      expect(error).toBeDefined();
      expect(chunks).toMatchInlineSnapshot(`
        [
          {
            "type": "start",
          },
          {
            "error": [Error: Some generic error],
            "type": "error",
          },
        ]
      `);
    });
  });

  describe('embed', () => {
    it('should succeed without errors', async () => {
      // Arrange
      const baseModel = new MockEmbeddingModel({ doEmbed: mockEmbeddings });
      const retryModel = new MockEmbeddingModel({ doEmbed: mockEmbeddings });

      // Act
      const result = await embed({
        model: createRetryable({
          model: baseModel,
          retries: [requestNotRetryable(retryModel)],
        }),
        value: 'Hello!',
      });

      // Assert
      expect(baseModel.doEmbed).toHaveBeenCalledTimes(1);
      expect(result.embedding).toEqual(mockEmbeddings.embeddings[0]);
    });

    it('should fallback in case of non-retryable error', async () => {
      // Arrange
      const baseModel = new MockEmbeddingModel({
        doEmbed: nonRetryableError,
      });
      const retryModel = new MockEmbeddingModel({ doEmbed: mockEmbeddings });

      // Act
      const result = await embed({
        model: createRetryable({
          model: baseModel,
          retries: [requestNotRetryable(retryModel)],
        }),
        value: 'Hello!',
        maxRetries: 0,
      });

      // Assert
      expect(baseModel.doEmbed).toHaveBeenCalledTimes(1);
      expect(retryModel.doEmbed).toHaveBeenCalledTimes(1);
      expect(result.embedding).toEqual(mockEmbeddings.embeddings[0]);
    });

    it('should not fallback for retryable error', async () => {
      // Arrange
      const baseModel = new MockEmbeddingModel({ doEmbed: retryableError });
      const retryModel = new MockEmbeddingModel({ doEmbed: mockEmbeddings });

      // Act
      const result = embed({
        model: createRetryable({
          model: baseModel,
          retries: [requestNotRetryable(retryModel)],
        }),
        value: 'Hello!',
        maxRetries: 0,
      });

      // Assert
      await expect(result).rejects.toThrowError(APICallError);
      expect(baseModel.doEmbed).toHaveBeenCalledTimes(1);
      expect(retryModel.doEmbed).toHaveBeenCalledTimes(0);
    });

    it('should not fallback for non AI SDK errors', async () => {
      // Arrange
      const genericError = new Error('Some generic error');
      const baseModel = new MockEmbeddingModel({ doEmbed: genericError });
      const retryModel = new MockEmbeddingModel({ doEmbed: mockEmbeddings });

      // Act
      const result = embed({
        model: createRetryable({
          model: baseModel,
          retries: [requestNotRetryable(retryModel)],
        }),
        value: 'Hello!',
        maxRetries: 0,
      });

      // Assert
      await expect(result).rejects.toThrowError(Error);
      expect(baseModel.doEmbed).toHaveBeenCalledTimes(1);
      expect(retryModel.doEmbed).toHaveBeenCalledTimes(0);
    });
  });

  describe('generateImage', () => {
    it('should succeed without errors', async () => {
      // Arrange
      const baseModel = new MockImageModel({ doGenerate: mockImageResult });
      const retryModel = new MockImageModel({ doGenerate: mockImageResult });

      // Act
      const result = await generateImage({
        model: createRetryable({
          model: baseModel,
          retries: [requestNotRetryable(retryModel)],
        }),
        prompt: 'test',
      });

      // Assert
      expect(baseModel.doGenerate).toHaveBeenCalledTimes(1);
      expect(retryModel.doGenerate).toHaveBeenCalledTimes(0);
      expect(result.images.length).toBe(1);
    });

    it('should fallback on non-retryable error', async () => {
      // Arrange
      const baseModel = new MockImageModel({ doGenerate: nonRetryableError });
      const retryModel = new MockImageModel({ doGenerate: mockImageResult });

      // Act
      const result = await generateImage({
        model: createRetryable({
          model: baseModel,
          retries: [requestNotRetryable(retryModel)],
        }),
        prompt: 'test',
        maxRetries: 0,
      });

      // Assert
      expect(baseModel.doGenerate).toHaveBeenCalledTimes(1);
      expect(retryModel.doGenerate).toHaveBeenCalledTimes(1);
      expect(result.images.length).toBe(1);
    });

    it('should not fallback for retryable errors', async () => {
      // Arrange
      const baseModel = new MockImageModel({ doGenerate: retryableError });
      const retryModel = new MockImageModel({ doGenerate: mockImageResult });

      // Act
      const result = generateImage({
        model: createRetryable({
          model: baseModel,
          retries: [requestNotRetryable(retryModel)],
        }),
        prompt: 'test',
        maxRetries: 0,
      });

      // Assert
      await expect(result).rejects.toThrowError(APICallError);
      expect(baseModel.doGenerate).toHaveBeenCalledTimes(1);
      expect(retryModel.doGenerate).toHaveBeenCalledTimes(0);
    });
  });
});
