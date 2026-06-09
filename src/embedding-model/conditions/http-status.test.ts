import { APICallError, embed } from 'ai';
import { describe, expect, it } from 'vitest';
import {
  apiError,
  createRetryableModel,
  MockEmbeddingModel,
} from '../../internal/test-utils.js';
import type { EmbeddingModelEmbed } from '../../types.js';
import { httpStatus } from './index.js';

const okEmbedding: EmbeddingModelEmbed = {
  embeddings: [[0.1, 0.2, 0.3]],
  warnings: [],
};

describe('httpStatus (embedding)', () => {
  describe('embed', () => {
    it('should switch on numeric status match', async () => {
      // Arrange
      const baseModel = new MockEmbeddingModel({
        doEmbed: apiError({ statusCode: 529 }),
      });
      const retryModel = new MockEmbeddingModel({ doEmbed: okEmbedding });

      // Act
      const result = await embed({
        model: createRetryableModel({
          model: baseModel,
          retries: [httpStatus(529).switch({ model: retryModel })],
        }),
        value: 'Hello!',
        maxRetries: 0,
      });

      // Assert
      expect(baseModel.doEmbed).toHaveBeenCalledTimes(1);
      expect(retryModel.doEmbed).toHaveBeenCalledTimes(1);
      expect(result.embedding).toEqual(okEmbedding.embeddings[0]);
    });

    it('should switch on message substring match', async () => {
      // Arrange
      const baseModel = new MockEmbeddingModel({
        doEmbed: apiError({
          statusCode: 200,
          message: 'Service Overloaded',
        }),
      });
      const retryModel = new MockEmbeddingModel({ doEmbed: okEmbedding });

      // Act
      const result = await embed({
        model: createRetryableModel({
          model: baseModel,
          retries: [httpStatus('overloaded').switch({ model: retryModel })],
        }),
        value: 'Hello!',
        maxRetries: 0,
      });

      // Assert
      expect(baseModel.doEmbed).toHaveBeenCalledTimes(1);
      expect(retryModel.doEmbed).toHaveBeenCalledTimes(1);
      expect(result.embedding).toEqual(okEmbedding.embeddings[0]);
    });

    it('should switch on regex match', async () => {
      // Arrange
      const baseModel = new MockEmbeddingModel({
        doEmbed: apiError({ statusCode: 502 }),
      });
      const retryModel = new MockEmbeddingModel({ doEmbed: okEmbedding });

      // Act
      const result = await embed({
        model: createRetryableModel({
          model: baseModel,
          retries: [httpStatus(/^5\d\d$/).switch({ model: retryModel })],
        }),
        value: 'Hello!',
        maxRetries: 0,
      });

      // Assert
      expect(baseModel.doEmbed).toHaveBeenCalledTimes(1);
      expect(retryModel.doEmbed).toHaveBeenCalledTimes(1);
      expect(result.embedding).toEqual(okEmbedding.embeddings[0]);
    });

    it('should not switch when no pattern matches', async () => {
      // Arrange
      const baseModel = new MockEmbeddingModel({
        doEmbed: apiError({ statusCode: 400, message: 'bad request' }),
      });
      const retryModel = new MockEmbeddingModel({ doEmbed: okEmbedding });

      // Act
      const result = embed({
        model: createRetryableModel({
          model: baseModel,
          retries: [
            httpStatus(529, 'overloaded').switch({ model: retryModel }),
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
});
