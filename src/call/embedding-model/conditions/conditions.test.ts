import { describe, expect, it } from 'vitest';
import {
  Embedding,
  MockEmbeddingModel,
  nonRetryableError,
  retryableError,
} from '../../../internal/test-utils.js';
import { retryableEmbed } from '../functions/embed.js';
import { retryableEmbedMany } from '../functions/embed-many.js';
import {
  error,
  httpStatus,
  isEmbedManyResult,
  isEmbedResult,
  or,
  result,
} from './index.js';

const value = 'Hello!';

describe('embedding-model call conditions', () => {
  describe('error', () => {
    it('should switch when the predicate matches', async () => {
      // Arrange
      const primary = MockEmbeddingModel.from(retryableError);
      const fallback = MockEmbeddingModel.from([Embedding.vector(3)]);

      // Act
      const out = await retryableEmbed({
        model: primary,
        value,
        retry: [error(() => true).switch({ model: fallback })],
      });

      // Assert
      expect(out.embedding.length).toBe(3);
      expect(fallback.doEmbed.mock.calls.length).toBe(1);
    });

    it('should not switch when the predicate misses', async () => {
      // Arrange
      const primary = MockEmbeddingModel.from(nonRetryableError);
      const fallback = MockEmbeddingModel.from([Embedding.vector(3)]);

      // Act
      const out = retryableEmbed({
        model: primary,
        value,
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
      await retryableEmbed({
        model: primary,
        value,
        retry: [
          or(httpStatus(503), httpStatus(429)).switch({ model: fallback }),
        ],
      });

      // Assert
      expect(fallback.doEmbed.mock.calls.length).toBe(1);
    });
  });

  describe('result', () => {
    it('should switch on a degenerate embedding', async () => {
      // Arrange — not an error, so nothing else could catch it.
      const primary = MockEmbeddingModel.from([[0, 0, 0]]);
      const fallback = MockEmbeddingModel.from([Embedding.vector(3)]);

      // Act
      const out = await retryableEmbed({
        model: primary,
        value,
        retry: [
          result(
            (res) => isEmbedResult(res) && res.embedding.every((n) => n === 0),
          ).switch({ model: fallback }),
        ],
      });

      // Assert
      expect(out.embedding).toEqual([0.1, 0.2, 0.3]);
      expect(fallback.doEmbed.mock.calls.length).toBe(1);
    });

    it('should switch on too few embeddings from embedMany', async () => {
      // Arrange
      const primary = MockEmbeddingModel.from([Embedding.vector(3)]);
      const fallback = MockEmbeddingModel.from([Embedding.vector(3)]);

      // Act
      await retryableEmbedMany({
        model: primary,
        values: [value],
        retry: [
          result(
            (res) => isEmbedManyResult(res) && res.embeddings.length < 2,
          ).switch({ model: fallback }),
        ],
      });

      // Assert
      expect(fallback.doEmbed.mock.calls.length).toBe(1);
    });

    it('should tag embed and embedMany apart from one condition instance', async () => {
      // Arrange — the same condition, reused across both entry points.
      const model = MockEmbeddingModel.from([Embedding.vector(3)]);
      const seen: Array<unknown> = [];
      const observe = result((res) => {
        seen.push({
          operation: res.operation,
          isEmbed: isEmbedResult(res),
          isEmbedMany: isEmbedManyResult(res),
        });
        return false;
      }).switch({ model });

      // Act
      await retryableEmbed({ model, value, retry: [observe] });
      await retryableEmbedMany({ model, values: [value], retry: [observe] });

      // Assert
      expect(seen).toEqual([
        { operation: 'embed', isEmbed: true, isEmbedMany: false },
        { operation: 'embedMany', isEmbed: false, isEmbedMany: true },
      ]);
    });

    it('should keep the result when the predicate misses', async () => {
      // Arrange
      const primary = MockEmbeddingModel.from([Embedding.vector(3)]);
      const fallback = MockEmbeddingModel.from([Embedding.vector(3)]);

      // Act
      await retryableEmbed({
        model: primary,
        value,
        retry: [result(() => false).switch({ model: fallback })],
      });

      // Assert
      expect(fallback.doEmbed.mock.calls.length).toBe(0);
    });
  });
});
