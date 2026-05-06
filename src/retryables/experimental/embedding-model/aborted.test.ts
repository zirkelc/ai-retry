import { embed } from 'ai';
import { describe, expect, it } from 'vitest';
import { createRetryable } from '../../../create-retryable-model.js';
import {
  abortError,
  MockEmbeddingModel,
  timeoutError,
} from '../../../test-utils.js';
import type { EmbeddingModelEmbed } from '../../../types.js';
import { aborted } from './index.js';

const okEmbedding: EmbeddingModelEmbed = {
  embeddings: [[0.1, 0.2, 0.3]],
  warnings: [],
};

describe('aborted (embedding)', () => {
  describe('embed', () => {
    it('should switch on AbortError with a fresh deadline', async () => {
      // Arrange
      const baseModel = new MockEmbeddingModel({ doEmbed: abortError() });
      const retryModel = new MockEmbeddingModel({ doEmbed: okEmbedding });

      // Act
      const result = await embed({
        model: createRetryable({
          model: baseModel,
          retries: [aborted().switch({ model: retryModel, timeout: 60_000 })],
        }),
        value: 'Hello!',
        maxRetries: 0,
      });

      // Assert
      expect(baseModel.doEmbed).toHaveBeenCalledTimes(1);
      expect(retryModel.doEmbed).toHaveBeenCalledTimes(1);
      expect(result.embedding).toEqual(okEmbedding.embeddings[0]);
    });

    it('should not switch on TimeoutError', async () => {
      // Arrange
      const baseModel = new MockEmbeddingModel({ doEmbed: timeoutError() });
      const retryModel = new MockEmbeddingModel({ doEmbed: okEmbedding });

      // Act
      const result = embed({
        model: createRetryable({
          model: baseModel,
          retries: [aborted().switch({ model: retryModel, timeout: 60_000 })],
        }),
        value: 'Hello!',
        maxRetries: 0,
      });

      // Assert
      await expect(result).rejects.toThrow(/timed out/);
      expect(baseModel.doEmbed).toHaveBeenCalledTimes(1);
      expect(retryModel.doEmbed).toHaveBeenCalledTimes(0);
    });
  });
});
