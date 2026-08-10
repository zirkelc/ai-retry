import { describe, expect, it } from 'vitest';
import {
  contentFilterStreamChunks,
  MockLanguageModel,
  mockStreamChunks,
  nonRetryableError,
  retryableError,
  Streams,
} from '../../../internal/test-utils.js';
import { retryableStreamText } from '../stream-text.js';
import {
  and,
  error,
  finishReason,
  httpStatus,
  not,
  or,
  result,
} from './index.js';

const prompt = 'Hello!';

describe('stream-text call conditions', () => {
  describe('error', () => {
    it('should switch when the predicate matches', async () => {
      // Arrange
      const primary = MockLanguageModel.from(retryableError);
      const fallback = MockLanguageModel.from({ doStream: mockStreamChunks });

      // Act
      const out = await retryableStreamText({
        model: primary,
        prompt,
        retry: [error(() => true).switch({ model: fallback })],
      });
      await Streams.toArray(out.fullStream);

      // Assert
      expect(fallback.doStream.mock.calls.length).toBe(1);
    });

    it('should not switch when the predicate misses', async () => {
      // Arrange
      const primary = MockLanguageModel.from(nonRetryableError);
      const fallback = MockLanguageModel.from({ doStream: mockStreamChunks });

      // Act
      const out = retryableStreamText({
        model: primary,
        prompt,
        retry: [error(() => false).switch({ model: fallback })],
      });

      // Assert
      await expect(out).rejects.toThrow();
      expect(fallback.doStream.mock.calls.length).toBe(0);
    });
  });

  describe('httpStatus', () => {
    it('should switch on a matching status code', async () => {
      // Arrange
      const primary = MockLanguageModel.from(retryableError);
      const fallback = MockLanguageModel.from({ doStream: mockStreamChunks });

      // Act
      const out = await retryableStreamText({
        model: primary,
        prompt,
        retry: [
          or(httpStatus(503), httpStatus(429)).switch({ model: fallback }),
        ],
      });
      await Streams.toArray(out.fullStream);

      // Assert
      expect(fallback.doStream.mock.calls.length).toBe(1);
    });
  });

  describe('finishReason', () => {
    it('should switch on a contentless content-filter finish', async () => {
      // Arrange — the motivating case: the stream ends without ever emitting a
      // content part, so the attempt has not committed and can still fail over.
      const primary = MockLanguageModel.from({
        doStream: contentFilterStreamChunks,
      });
      const fallback = MockLanguageModel.from({ doStream: mockStreamChunks });

      // Act
      const out = await retryableStreamText({
        model: primary,
        prompt,
        retry: [finishReason('content-filter').switch({ model: fallback })],
      });
      await Streams.toArray(out.fullStream);

      // Assert
      expect(fallback.doStream.mock.calls.length).toBe(1);
    });

    it('should not switch on a different reason', async () => {
      // Arrange
      const primary = MockLanguageModel.from({
        doStream: contentFilterStreamChunks,
      });
      const fallback = MockLanguageModel.from({ doStream: mockStreamChunks });

      // Act
      const out = await retryableStreamText({
        model: primary,
        prompt,
        retry: [finishReason('length').switch({ model: fallback })],
      });
      await Streams.toArray(out.fullStream);

      // Assert
      expect(fallback.doStream.mock.calls.length).toBe(0);
    });
  });

  describe('result', () => {
    it('should judge what a pre-commit stream reports', async () => {
      // Arrange — no content exists to read, so a condition here is written
      // against the finish, the usage and the provider metadata instead. The
      // last is the one `finishReason` cannot reach.
      const primary = MockLanguageModel.from({
        doStream: contentFilterStreamChunks,
      });
      const fallback = MockLanguageModel.from({ doStream: mockStreamChunks });

      // Act
      const out = await retryableStreamText({
        model: primary,
        prompt,
        retry: [
          result(
            (res) =>
              res.usage.inputTokens === 3 &&
              res.providerMetadata?.testProvider?.testKey === 'testValue',
          ).switch({ model: fallback }),
        ],
      });
      await Streams.toArray(out.fullStream);

      // Assert
      expect(fallback.doStream.mock.calls.length).toBe(1);
    });

    it('should keep the stream when the predicate misses', async () => {
      // Arrange
      const primary = MockLanguageModel.from({
        doStream: contentFilterStreamChunks,
      });
      const fallback = MockLanguageModel.from({ doStream: mockStreamChunks });

      // Act
      const out = await retryableStreamText({
        model: primary,
        prompt,
        retry: [result(() => false).switch({ model: fallback })],
      });
      await Streams.toArray(out.fullStream);

      // Assert
      expect(fallback.doStream.mock.calls.length).toBe(0);
    });
  });

  describe('combinators', () => {
    it('should require both sides for and()', async () => {
      // Arrange
      const primary = MockLanguageModel.from(retryableError);
      const fallback = MockLanguageModel.from({ doStream: mockStreamChunks });

      // Act
      const out = retryableStreamText({
        model: primary,
        prompt,
        retry: [
          and(httpStatus(429), httpStatus(503)).switch({ model: fallback }),
        ],
      });

      // Assert
      await expect(out).rejects.toThrow();
      expect(fallback.doStream.mock.calls.length).toBe(0);
    });

    it('should invert with not()', async () => {
      // Arrange
      const primary = MockLanguageModel.from(retryableError);
      const fallback = MockLanguageModel.from({ doStream: mockStreamChunks });

      // Act
      const out = await retryableStreamText({
        model: primary,
        prompt,
        retry: [not(httpStatus(503)).switch({ model: fallback })],
      });
      await Streams.toArray(out.fullStream);

      // Assert
      expect(fallback.doStream.mock.calls.length).toBe(1);
    });
  });
});
