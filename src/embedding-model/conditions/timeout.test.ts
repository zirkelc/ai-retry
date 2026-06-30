import { embed } from 'ai';
import { describe, expect, it } from 'vitest';
import {
  Errors,
  MockEmbeddingModel,
  createRetryableModel,
} from '../../internal/test-utils.js';
import type { EmbeddingModelEmbed } from '../../types.js';
import { timeout } from './index.js';

const okEmbedding: EmbeddingModelEmbed = {
  embeddings: [[0.1, 0.2, 0.3]],
  warnings: [],
};

describe('timeout (embedding)', () => {
  describe('embed', () => {
    it('should switch on TimeoutError', async () => {
      // Arrange
      const baseModel = MockEmbeddingModel.from(Errors.timeout());
      const retryModel = MockEmbeddingModel.from(okEmbedding);

      // Act
      const result = await embed({
        model: createRetryableModel({
          model: baseModel,
          retries: [timeout().switch({ model: retryModel })],
        }),
        value: 'Hello!',
        maxRetries: 0,
      });

      // Assert
      expect(baseModel.doEmbed).toHaveBeenCalledTimes(1);
      expect(retryModel.doEmbed).toHaveBeenCalledTimes(1);
      expect(result.embedding).toEqual(okEmbedding.embeddings[0]);
    });

    it('should not switch on AbortError', async () => {
      // Arrange
      const baseModel = MockEmbeddingModel.from(Errors.abort());
      const retryModel = MockEmbeddingModel.from(okEmbedding);

      // Act
      const result = embed({
        model: createRetryableModel({
          model: baseModel,
          retries: [timeout().switch({ model: retryModel })],
        }),
        value: 'Hello!',
        maxRetries: 0,
      });

      // Assert
      await expect(result).rejects.toThrow(/aborted/);
      expect(baseModel.doEmbed).toHaveBeenCalledTimes(1);
      expect(retryModel.doEmbed).toHaveBeenCalledTimes(0);
    });
  });
});
