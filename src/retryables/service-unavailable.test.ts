import { Errors, Iterables } from 'ai-test-kit';
import {
  APICallError,
  embed,
  generateImage,
  generateText,
  streamText,
} from 'ai';
import { describe, expect, it } from 'vitest';
import {
  MockEmbeddingModel,
  MockImageModel,
  MockLanguageModel,
  chunksToText,
  createRetryableModel,
  mockEmbeddings,
  mockImageResult,
  mockResult,
  mockResultText,
  mockStreamChunks,
} from '../internal/test-utils.js';
import { serviceUnavailable } from './service-unavailable.js';

const unavailableError = Errors.serviceUnavailable();

describe('serviceUnavailable', () => {
  describe('generateText', () => {
    it('should succeed without errors', async () => {
      // Arrange
      const baseModel = MockLanguageModel.from({ doGenerate: mockResult });
      const retryModel = MockLanguageModel.from({ doGenerate: mockResult });

      // Act
      const result = await generateText({
        model: createRetryableModel({
          model: baseModel,
          retries: [serviceUnavailable(retryModel)],
        }),
        prompt: 'Hello!',
      });

      // Assert
      expect(baseModel.doGenerate).toHaveBeenCalledTimes(1);
      expect(result.text).toBe(mockResultText);
    });

    it('should retry for status 503', async () => {
      // Arrange
      const baseModel = MockLanguageModel.from({
        doGenerate: unavailableError,
      });
      const retryModel = MockLanguageModel.from({ doGenerate: mockResult });

      // Act
      const result = await generateText({
        model: createRetryableModel({
          model: baseModel,
          retries: [serviceUnavailable(retryModel)],
        }),
        prompt: 'Hello!',
      });

      // Assert
      expect(baseModel.doGenerate).toHaveBeenCalledTimes(1);
      expect(retryModel.doGenerate).toHaveBeenCalledTimes(1);
      expect(result.text).toBe(mockResultText);
    });

    it('should not retry for status 200', async () => {
      // Arrange
      const baseModel = MockLanguageModel.from({ doGenerate: mockResult });
      const retryModel = MockLanguageModel.from({ doGenerate: mockResult });

      // Act
      const result = await generateText({
        model: createRetryableModel({
          model: baseModel,
          retries: [serviceUnavailable(retryModel)],
        }),
        prompt: 'Hello!',
      });

      // Assert
      expect(baseModel.doGenerate).toHaveBeenCalledTimes(1);
      expect(retryModel.doGenerate).toHaveBeenCalledTimes(0);
      expect(result.text).toBe(mockResultText);
    });

    it('should not retry if no matches', async () => {
      // Arrange
      const baseModel = MockLanguageModel.from({
        doGenerate: Errors.badRequest(),
      });

      const retryModel = MockLanguageModel.from({ doGenerate: mockResult });

      // Act
      const result = generateText({
        model: createRetryableModel({
          model: baseModel,
          retries: [serviceUnavailable(retryModel)],
        }),
        prompt: 'Hello!',
        maxRetries: 0,
      });

      // Assert
      await expect(result).rejects.toThrowError(APICallError);
      expect(baseModel.doGenerate).toHaveBeenCalledTimes(1);
      expect(retryModel.doGenerate).toHaveBeenCalledTimes(0);
    });
  });

  describe('streamText', () => {
    it('should succeed without errors', async () => {
      // Arrange
      const baseModel = MockLanguageModel.from({
        doStream: mockStreamChunks,
      });
      const retryModel = MockLanguageModel.from({
        doStream: mockStreamChunks,
      });
      let error: unknown;

      // Act
      const result = streamText({
        model: createRetryableModel({
          model: baseModel,
          retries: [serviceUnavailable(retryModel)],
        }),
        prompt: 'Hello!',
        onError(data) {
          error = data.error;
        },
      });

      const chunks = await Iterables.toArray(result.fullStream);

      // Assert
      expect(baseModel.doStream).toHaveBeenCalledTimes(1);
      expect(retryModel.doStream).toHaveBeenCalledTimes(0);
      expect(error).toBeUndefined();
      expect(chunksToText(chunks)).toBe(mockResultText);
    });

    it('should retry for status 503', async () => {
      // Arrange
      const baseModel = MockLanguageModel.from({ doStream: unavailableError });
      const retryModel = MockLanguageModel.from({
        doStream: mockStreamChunks,
      });
      let error: unknown;

      // Act
      const result = streamText({
        model: createRetryableModel({
          model: baseModel,
          retries: [serviceUnavailable(retryModel)],
        }),
        prompt: 'Hello!',
        onError(data) {
          error = data.error;
        },
      });

      const chunks = await Iterables.toArray(result.fullStream);

      // Assert
      expect(baseModel.doStream).toHaveBeenCalledTimes(1);
      expect(retryModel.doStream).toHaveBeenCalledTimes(1);
      expect(error).toBeUndefined();
      expect(chunksToText(chunks)).toBe(mockResultText);
    });

    it('should not retry if no matches', async () => {
      // Arrange
      const baseModel = MockLanguageModel.from({
        doStream: Errors.badRequest(),
      });

      const retryModel = MockLanguageModel.from({
        doStream: mockStreamChunks,
      });
      let error: unknown;

      // Act
      const result = streamText({
        model: createRetryableModel({
          model: baseModel,
          retries: [serviceUnavailable(retryModel)],
        }),
        prompt: 'Hello!',
        maxRetries: 0,
        onError(data) {
          error = data.error;
        },
      });

      const chunks = await Iterables.toArray(result.fullStream);

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
            "error": [AI_APICallError: Bad request],
            "type": "error",
          },
        ]
      `);
    });
  });

  describe('embed', () => {
    it('should succeed without errors', async () => {
      // Arrange
      const baseModel = MockEmbeddingModel.from(mockEmbeddings);
      const retryModel = MockEmbeddingModel.from(mockEmbeddings);

      // Act
      const result = await embed({
        model: createRetryableModel({
          model: baseModel,
          retries: [serviceUnavailable(retryModel)],
        }),
        value: 'Hello!',
      });

      // Assert
      expect(baseModel.doEmbed).toHaveBeenCalledTimes(1);
      expect(result.embedding).toBe(mockEmbeddings.embeddings[0]);
    });

    it('should retry for status 503', async () => {
      // Arrange
      const baseModel = MockEmbeddingModel.from(unavailableError);
      const retryModel = MockEmbeddingModel.from(mockEmbeddings);

      // Act
      const result = await embed({
        model: createRetryableModel({
          model: baseModel,
          retries: [serviceUnavailable(retryModel)],
        }),
        value: 'Hello!',
      });

      // Assert
      expect(baseModel.doEmbed).toHaveBeenCalledTimes(1);
      expect(retryModel.doEmbed).toHaveBeenCalledTimes(1);
      expect(result.embedding).toBe(mockEmbeddings.embeddings[0]);
    });

    it('should not retry for status 200', async () => {
      // Arrange
      const baseModel = MockEmbeddingModel.from(mockEmbeddings);
      const retryModel = MockEmbeddingModel.from(mockEmbeddings);

      // Act
      const result = await embed({
        model: createRetryableModel({
          model: baseModel,
          retries: [serviceUnavailable(retryModel)],
        }),
        value: 'Hello!',
      });

      // Assert
      expect(baseModel.doEmbed).toHaveBeenCalledTimes(1);
      expect(retryModel.doEmbed).toHaveBeenCalledTimes(0);
      expect(result.embedding).toBe(mockEmbeddings.embeddings[0]);
    });

    it('should not retry if no matches', async () => {
      // Arrange
      const baseModel = MockEmbeddingModel.from(Errors.badRequest());

      const retryModel = MockEmbeddingModel.from(mockEmbeddings);

      // Act
      const result = embed({
        model: createRetryableModel({
          model: baseModel,
          retries: [serviceUnavailable(retryModel)],
        }),
        value: 'Hello!',
        maxRetries: 0,
      });

      // Assert
      await expect(result).rejects.toThrowError(APICallError);
      expect(baseModel.doEmbed).toHaveBeenCalledTimes(1);
      expect(retryModel.doEmbed).toHaveBeenCalledTimes(0);
    });
  });

  describe('generateImage', () => {
    it('should succeed without errors', async () => {
      // Arrange
      const baseModel = MockImageModel.from(mockImageResult);
      const retryModel = MockImageModel.from(mockImageResult);

      // Act
      const result = await generateImage({
        model: createRetryableModel({
          model: baseModel,
          retries: [serviceUnavailable(retryModel)],
        }),
        prompt: 'test',
      });

      // Assert
      expect(baseModel.doGenerate).toHaveBeenCalledTimes(1);
      expect(retryModel.doGenerate).toHaveBeenCalledTimes(0);
      expect(result.images.length).toBe(1);
    });

    it('should fallback on status 503', async () => {
      // Arrange
      const baseModel = MockImageModel.from(unavailableError);
      const retryModel = MockImageModel.from(mockImageResult);

      // Act
      const result = await generateImage({
        model: createRetryableModel({
          model: baseModel,
          retries: [serviceUnavailable(retryModel)],
        }),
        prompt: 'test',
      });

      // Assert
      expect(baseModel.doGenerate).toHaveBeenCalledTimes(1);
      expect(retryModel.doGenerate).toHaveBeenCalledTimes(1);
      expect(result.images.length).toBe(1);
    });

    it('should not fallback for non-matching errors', async () => {
      // Arrange
      const baseModel = MockImageModel.from(Errors.badRequest());
      const retryModel = MockImageModel.from(mockImageResult);

      // Act
      const result = generateImage({
        model: createRetryableModel({
          model: baseModel,
          retries: [serviceUnavailable(retryModel)],
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
