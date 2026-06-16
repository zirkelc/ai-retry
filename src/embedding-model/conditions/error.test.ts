import { APICallError, embed } from 'ai';
import { describe, expect, it } from 'vitest';
import {
  apiError,
  createRetryableModel,
  MockEmbeddingModel,
} from '../../internal/test-utils.js';
import type { EmbeddingModelEmbed } from '../../types.js';
import { error } from './index.js';

const okEmbedding: EmbeddingModelEmbed = {
  embeddings: [[0.1, 0.2, 0.3]],
  warnings: [],
};

describe('error (embedding)', () => {
  describe('embed', () => {
    it('should switch when predicate matches the error', async () => {
      // Arrange
      const baseModel = new MockEmbeddingModel({
        doEmbed: apiError({ statusCode: 418 }),
      });
      const retryModel = new MockEmbeddingModel({ doEmbed: okEmbedding });

      // Act
      const result = await embed({
        model: createRetryableModel({
          model: baseModel,
          retries: [
            error(
              (e) => APICallError.isInstance(e) && e.statusCode === 418,
            ).switch({ model: retryModel }),
          ],
        }),
        value: 'Hello!',
        maxRetries: 0,
      });

      // Assert
      expect(baseModel.doEmbed).toHaveBeenCalledTimes(1);
      expect(retryModel.doEmbed).toHaveBeenCalledTimes(1);
      expect(result.embedding).toEqual(okEmbedding.embeddings[0]);
    });

    it('should not switch when predicate misses', async () => {
      // Arrange
      const baseModel = new MockEmbeddingModel({
        doEmbed: apiError({ statusCode: 500 }),
      });
      const retryModel = new MockEmbeddingModel({ doEmbed: okEmbedding });

      // Act
      const result = embed({
        model: createRetryableModel({
          model: baseModel,
          retries: [
            error(
              (e) => APICallError.isInstance(e) && e.statusCode === 418,
            ).switch({ model: retryModel }),
          ],
        }),
        value: 'Hello!',
        maxRetries: 0,
      });

      // Assert
      await expect(result).rejects.toThrow(APICallError);
      expect(baseModel.doEmbed).toHaveBeenCalledTimes(1);
      expect(retryModel.doEmbed).toHaveBeenCalledTimes(0);
    });
  });

  describe('error.isRetryable', () => {
    it('should retry the same model when isRetryable=true', async () => {
      // Arrange
      let attempt = 0;
      const baseModel = new MockEmbeddingModel({
        doEmbed: async () => {
          attempt++;
          if (attempt === 1) {
            throw apiError({ statusCode: 503, isRetryable: true });
          }
          return okEmbedding;
        },
      });

      // Act
      const result = await embed({
        model: createRetryableModel({
          model: baseModel,
          retries: [error.isRetryable(true).retry()],
        }),
        value: 'Hello!',
        maxRetries: 0,
      });

      // Assert
      expect(baseModel.doEmbed).toHaveBeenCalledTimes(2);
      expect(result.embedding).toEqual(okEmbedding.embeddings[0]);
    });

    it('should switch when isRetryable=false', async () => {
      // Arrange
      const baseModel = new MockEmbeddingModel({
        doEmbed: apiError({ statusCode: 400, isRetryable: false }),
      });
      const retryModel = new MockEmbeddingModel({ doEmbed: okEmbedding });

      // Act
      const result = await embed({
        model: createRetryableModel({
          model: baseModel,
          retries: [error.isRetryable(false).switch({ model: retryModel })],
        }),
        value: 'Hello!',
        maxRetries: 0,
      });

      // Assert
      expect(baseModel.doEmbed).toHaveBeenCalledTimes(1);
      expect(retryModel.doEmbed).toHaveBeenCalledTimes(1);
      expect(result.embedding).toEqual(okEmbedding.embeddings[0]);
    });
  });

  describe('error.statusCode', () => {
    it('should switch on matching numeric status', async () => {
      // Arrange
      const baseModel = new MockEmbeddingModel({
        doEmbed: apiError({ statusCode: 503 }),
      });
      const retryModel = new MockEmbeddingModel({ doEmbed: okEmbedding });

      // Act
      const result = await embed({
        model: createRetryableModel({
          model: baseModel,
          retries: [error.statusCode(503).switch({ model: retryModel })],
        }),
        value: 'Hello!',
        maxRetries: 0,
      });

      // Assert
      expect(baseModel.doEmbed).toHaveBeenCalledTimes(1);
      expect(retryModel.doEmbed).toHaveBeenCalledTimes(1);
      expect(result.embedding).toEqual(okEmbedding.embeddings[0]);
    });
  });

  describe('error.message', () => {
    it('should switch on case-insensitive substring match', async () => {
      // Arrange
      const baseModel = new MockEmbeddingModel({
        doEmbed: apiError({ message: 'Service Overloaded' }),
      });
      const retryModel = new MockEmbeddingModel({ doEmbed: okEmbedding });

      // Act
      const result = await embed({
        model: createRetryableModel({
          model: baseModel,
          retries: [error.message('overloaded').switch({ model: retryModel })],
        }),
        value: 'Hello!',
        maxRetries: 0,
      });

      // Assert
      expect(baseModel.doEmbed).toHaveBeenCalledTimes(1);
      expect(retryModel.doEmbed).toHaveBeenCalledTimes(1);
      expect(result.embedding).toEqual(okEmbedding.embeddings[0]);
    });
  });
});
