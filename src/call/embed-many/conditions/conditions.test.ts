import { describe, expect, it } from 'vitest';
import {
  Embedding,
  MockEmbeddingModel,
  nonRetryableError,
  retryableError,
} from '../../../internal/test-utils.js';
import { retryableEmbedMany } from '../embed-many.js';
import { error, httpStatus, or, result } from './index.js';

const values = ['Hello!'];

describe('embed-many call conditions', () => {
  describe('error', () => {
    it('should switch when the predicate matches', async () => {
      // Arrange
      const primary = MockEmbeddingModel.from(retryableError);
      const fallback = MockEmbeddingModel.from([Embedding.vector(3)]);

      // Act
      const out = await retryableEmbedMany({
        model: primary,
        values,
        retry: [error(() => true).switch({ model: fallback })],
      });

      // Assert
      expect(out.embeddings.length).toBe(1);
      expect(fallback.doEmbed.mock.calls.length).toBe(1);
    });

    it('should not switch when the predicate misses', async () => {
      // Arrange
      const primary = MockEmbeddingModel.from(nonRetryableError);
      const fallback = MockEmbeddingModel.from([Embedding.vector(3)]);

      // Act
      const out = retryableEmbedMany({
        model: primary,
        values,
        retry: [error(() => false).switch({ model: fallback })],
      });

      // Assert
      await expect(out).rejects.toThrow();
      expect(fallback.doEmbed.mock.calls.length).toBe(0);
    });
  });

  describe('httpStatus', () => {
    it('should switch on a matching status code', async () => {
      // Arrange
      const primary = MockEmbeddingModel.from(retryableError);
      const fallback = MockEmbeddingModel.from([Embedding.vector(3)]);

      // Act
      await retryableEmbedMany({
        model: primary,
        values,
        retry: [
          or(httpStatus(503), httpStatus(429)).switch({ model: fallback }),
        ],
      });

      // Assert
      expect(fallback.doEmbed.mock.calls.length).toBe(1);
    });
  });

  describe('result', () => {
    it('should switch on too few embeddings', async () => {
      // Arrange — the plural field, which is what separates this entry point's
      // conditions from `embed`'s. Not an error, so nothing else could catch it.
      const primary = MockEmbeddingModel.from([Embedding.vector(3)]);
      const fallback = MockEmbeddingModel.from([Embedding.vector(3)]);

      // Act
      await retryableEmbedMany({
        model: primary,
        values,
        retry: [
          result((res) => res.embeddings.length < 2).switch({
            model: fallback,
          }),
        ],
      });

      // Assert
      expect(fallback.doEmbed.mock.calls.length).toBe(1);
    });

    it('should keep the result when the predicate misses', async () => {
      // Arrange
      const primary = MockEmbeddingModel.from([Embedding.vector(3)]);
      const fallback = MockEmbeddingModel.from([Embedding.vector(3)]);

      // Act
      await retryableEmbedMany({
        model: primary,
        values,
        retry: [result(() => false).switch({ model: fallback })],
      });

      // Assert
      expect(fallback.doEmbed.mock.calls.length).toBe(0);
    });
  });
});
