import { describe, expect, it } from 'vitest';
import {
  Embedding,
  MockEmbeddingModel,
  nonRetryableError,
  retryableError,
} from '../../../internal/test-utils.js';
import type { EmbeddingModel } from '../../../types.js';
import {
  isEmbedManyResult,
  result as embeddingResult,
} from '../conditions/index.js';
import { retryableEmbedMany } from './embed-many.js';

const values = ['hi', 'there'];

/** A `doEmbed` that takes `ms` to answer, and rejects if aborted first. */
const slowEmbed =
  (ms: number): EmbeddingModel['doEmbed'] =>
  async ({ abortSignal }) => {
    await new Promise<void>((resolve, reject) => {
      const handle = setTimeout(resolve, ms);
      abortSignal?.addEventListener('abort', () => {
        clearTimeout(handle);
        reject(abortSignal.reason);
      });
    });
    return Embedding.result([Embedding.vector(3)]);
  };

describe('retryableEmbedMany', () => {
  describe('success', () => {
    it('should return the first attempt when nothing fails', async () => {
      // Arrange
      const model = MockEmbeddingModel.from([Embedding.vector(3)]);

      // Act
      const result = await retryableEmbedMany({ model, values });

      // Assert
      expect(result.embeddings.length).toBe(2);
      expect(result.values).toEqual(values);
    });

    it('should hand back the SDK result untouched', async () => {
      // Arrange
      const model = MockEmbeddingModel.from([Embedding.vector(3)]);

      // Act
      const result = await retryableEmbedMany({ model, values });

      // Assert
      expect('operation' in result).toBe(false);
    });
  });

  describe('retries', () => {
    it('should fall over to the next model after an error', async () => {
      // Arrange
      const primary = MockEmbeddingModel.from(retryableError);
      const fallback = MockEmbeddingModel.from([Embedding.vector(3)]);

      // Act
      const result = await retryableEmbedMany({
        model: primary,
        values,
        retry: [fallback],
      });

      // Assert — a retry re-runs the whole call, so the fallback embeds both
      // values (one batch each, at this mock's `maxEmbeddingsPerCall` of 1)
      // rather than only the batch that failed.
      expect(result.embeddings.length).toBe(2);
      expect(fallback.doEmbed.mock.calls.length).toBe(2);
    });

    it('should surface the error when no retry matched', async () => {
      // Arrange
      const primary = MockEmbeddingModel.from(nonRetryableError);

      // Act
      const result = retryableEmbedMany({ model: primary, values, retry: [] });

      // Assert
      await expect(result).rejects.toThrow(nonRetryableError);
    });
  });

  describe('deadlines', () => {
    it('should compose a retry timeout into the abort signal', async () => {
      // Arrange — like `embed`, there is no `timeout` argument to use.
      const primary = MockEmbeddingModel.from(retryableError);
      const slow = MockEmbeddingModel.from(slowEmbed(5_000));
      const rescue = MockEmbeddingModel.from([Embedding.vector(3)]);

      // Act
      const result = await retryableEmbedMany({
        model: primary,
        values: ['hi'],
        retry: [{ model: slow, timeout: 50 }, rescue],
      });

      // Assert
      expect(result.embeddings.length).toBe(1);
      expect(primary.doEmbed.mock.calls[0]![0].abortSignal).toBeUndefined();
      expect(slow.doEmbed.mock.calls[0]![0].abortSignal).toBeDefined();
      expect(rescue.doEmbed.mock.calls.length).toBe(1);
    });
  });

  describe('argument overrides', () => {
    it('should override the values for the retry attempt', async () => {
      // Arrange
      const primary = MockEmbeddingModel.from(retryableError);
      const fallback = MockEmbeddingModel.from([Embedding.vector(3)]);

      // Act
      await retryableEmbedMany({
        model: primary,
        values,
        retry: [{ model: fallback, options: { values: ['rephrased'] } }],
      });

      // Assert — one value now, so one batch.
      expect(fallback.doEmbed.mock.calls.length).toBe(1);
      expect(fallback.doEmbed.mock.calls[0]![0].values).toEqual(['rephrased']);
    });
  });

  describe('result-based retries', () => {
    it('should fall over on too few embeddings', async () => {
      // Arrange
      const primary = MockEmbeddingModel.from([Embedding.vector(3)]);
      const fallback = MockEmbeddingModel.from([Embedding.vector(3)]);

      // Act
      await retryableEmbedMany({
        model: primary,
        values: ['hi'],
        retry: [
          embeddingResult(
            (res) => isEmbedManyResult(res) && res.embeddings.length < 2,
          ).switch({ model: fallback }),
        ],
      });

      // Assert
      expect(fallback.doEmbed.mock.calls.length).toBe(1);
    });

    it('should keep the result when no condition matches', async () => {
      // Arrange
      const primary = MockEmbeddingModel.from([Embedding.vector(3)]);
      const fallback = MockEmbeddingModel.from([Embedding.vector(3)]);

      // Act
      const result = await retryableEmbedMany({
        model: primary,
        values: ['hi'],
        retry: [embeddingResult(() => false).switch({ model: fallback })],
      });

      // Assert
      expect(result.embeddings.length).toBe(1);
      expect(fallback.doEmbed.mock.calls.length).toBe(0);
    });
  });
});
