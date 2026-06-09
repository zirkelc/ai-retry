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
import {
  chunksToText,
  createRetryableModel,
  MockEmbeddingModel,
  mockEmbeddings,
  MockImageModel,
  mockImageResult,
  MockLanguageModel,
  mockResult,
  mockResultText,
  mockStreamChunks,
  nonRetryableError,
  retryableError,
} from '../internal/test-utils.js';
import { requestNotRetryable } from './request-not-retryable.js';

describe('requestNotRetryable', () => {
  describe('generateText', () => {
    it('should succeed without errors', async () => {
      // Arrange
      const baseModel = new MockLanguageModel({ doGenerate: mockResult });
      const retryModel = new MockLanguageModel({ doGenerate: mockResult });

      // Act
      const result = await generateText({
        model: createRetryableModel({
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
        model: createRetryableModel({
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
        model: createRetryableModel({
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
        model: createRetryableModel({
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
        model: createRetryableModel({
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
        model: createRetryableModel({
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
        model: createRetryableModel({
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
        model: createRetryableModel({
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
        model: createRetryableModel({
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
        model: createRetryableModel({
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
        model: createRetryableModel({
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
        model: createRetryableModel({
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
        model: createRetryableModel({
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
        model: createRetryableModel({
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
        model: createRetryableModel({
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
