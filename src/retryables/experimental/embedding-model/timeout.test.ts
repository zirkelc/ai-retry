import { embed } from 'ai';
import { describe, expect, it } from 'vitest';
import { createRetryable } from '../../../create-retryable-model.js';
import {
  abortError,
  MockEmbeddingModel,
  timeoutError,
} from '../../../test-utils.js';
import type { EmbeddingModelEmbed } from '../../../types.js';
import { timeout } from './index.js';

const okEmbedding: EmbeddingModelEmbed = {
  embeddings: [[0.1, 0.2, 0.3]],
  warnings: [],
};

describe('timeout (embedding)', () => {
  describe('embed', () => {
    it('should switch on TimeoutError', async () => {
      // Arrange
      const baseModel = new MockEmbeddingModel({ doEmbed: timeoutError() });
      const retryModel = new MockEmbeddingModel({ doEmbed: okEmbedding });

      // Act
      const result = await embed({
        model: createRetryable({
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
      const baseModel = new MockEmbeddingModel({ doEmbed: abortError() });
      const retryModel = new MockEmbeddingModel({ doEmbed: okEmbedding });

      // Act
      const result = embed({
        model: createRetryable({
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
