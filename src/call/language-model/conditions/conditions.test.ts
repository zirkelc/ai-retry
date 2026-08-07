import { describe, expect, it } from 'vitest';
import {
  contentFilterStreamChunks,
  MockLanguageModel,
  mockResultText,
  mockStreamChunks,
  nonRetryableError,
  retryableError,
  Streams,
} from '../../../internal/test-utils.js';
import { retryableGenerateText } from '../functions/generate-text.js';
import { retryableStreamText } from '../functions/stream-text.js';
import {
  and,
  error,
  finishReason,
  httpStatus,
  isGenerateTextResult,
  not,
  or,
  result,
} from './index.js';

const prompt = 'Hello!';

describe('language-model call conditions', () => {
  describe('error', () => {
    it('should switch when the predicate matches', async () => {
      // Arrange
      const primary = MockLanguageModel.from(retryableError);
      const fallback = MockLanguageModel.from(mockResultText);

      // Act
      const out = await retryableGenerateText({
        model: primary,
        prompt,
        retry: [error(() => true).switch({ model: fallback })],
      });

      // Assert
      expect(out.text).toBe(mockResultText);
      expect(primary.doGenerate.mock.calls.length).toBe(1);
      expect(fallback.doGenerate.mock.calls.length).toBe(1);
    });

    it('should not switch when the predicate misses', async () => {
      // Arrange
      const primary = MockLanguageModel.from(nonRetryableError);
      const fallback = MockLanguageModel.from(mockResultText);

      // Act
      const out = retryableGenerateText({
        model: primary,
        prompt,
        retry: [error(() => false).switch({ model: fallback })],
      });

      // Assert
      await expect(out).rejects.toThrow();
      expect(fallback.doGenerate.mock.calls.length).toBe(0);
    });
  });

  describe('httpStatus', () => {
    it('should switch on a matching status code', async () => {
      // Arrange
      const primary = MockLanguageModel.from(retryableError);
      const fallback = MockLanguageModel.from(mockResultText);

      // Act
      const out = await retryableGenerateText({
        model: primary,
        prompt,
        retry: [httpStatus(429).switch({ model: fallback })],
      });

      // Assert
      expect(out.text).toBe(mockResultText);
      expect(fallback.doGenerate.mock.calls.length).toBe(1);
    });

    it('should retry the same model rather than switch', async () => {
      // Arrange — one failure, then a success from the same model.
      const model = MockLanguageModel.from([retryableError, mockResultText]);

      // Act
      const out = await retryableGenerateText({
        model,
        prompt,
        retry: [httpStatus(429).retry({ maxAttempts: 2 })],
      });

      // Assert
      expect(out.text).toBe(mockResultText);
      expect(model.doGenerate.mock.calls.length).toBe(2);
    });
  });

  describe('finishReason', () => {
    it('should switch on a content-filter finish', async () => {
      // Arrange
      const primary = MockLanguageModel.from({
        content: [],
        finishReason: 'content-filter',
      });
      const fallback = MockLanguageModel.from(mockResultText);

      // Act
      const out = await retryableGenerateText({
        model: primary,
        prompt,
        retry: [finishReason('content-filter').switch({ model: fallback })],
      });

      // Assert
      expect(out.text).toBe(mockResultText);
      expect(fallback.doGenerate.mock.calls.length).toBe(1);
    });

    it('should switch on a contentless stream finish', async () => {
      // Arrange
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
  });

  describe('result', () => {
    it('should switch on the generated text', async () => {
      // Arrange
      const primary = MockLanguageModel.from('too short');
      const fallback = MockLanguageModel.from(mockResultText);

      // Act
      const out = await retryableGenerateText({
        model: primary,
        prompt,
        retry: [
          result(
            (res) => isGenerateTextResult(res) && res.text.length < 10,
          ).switch({ model: fallback }),
        ],
      });

      // Assert
      expect(out.text).toBe(mockResultText);
      expect(fallback.doGenerate.mock.calls.length).toBe(1);
    });

    it('should keep the result when the predicate misses', async () => {
      // Arrange
      const primary = MockLanguageModel.from(mockResultText);
      const fallback = MockLanguageModel.from('other');

      // Act
      const out = await retryableGenerateText({
        model: primary,
        prompt,
        retry: [result(() => false).switch({ model: fallback })],
      });

      // Assert
      expect(out.text).toBe(mockResultText);
      expect(fallback.doGenerate.mock.calls.length).toBe(0);
    });
  });

  describe('combinators', () => {
    it('should switch when or() matches either side', async () => {
      // Arrange
      const primary = MockLanguageModel.from(retryableError);
      const fallback = MockLanguageModel.from(mockResultText);

      // Act
      const out = await retryableGenerateText({
        model: primary,
        prompt,
        retry: [
          or(httpStatus(503), httpStatus(429)).switch({ model: fallback }),
        ],
      });

      // Assert
      expect(out.text).toBe(mockResultText);
      expect(fallback.doGenerate.mock.calls.length).toBe(1);
    });

    it('should require both sides for and()', async () => {
      // Arrange
      const primary = MockLanguageModel.from(retryableError);
      const fallback = MockLanguageModel.from(mockResultText);

      // Act
      const out = retryableGenerateText({
        model: primary,
        prompt,
        retry: [
          and(httpStatus(429), httpStatus(503)).switch({ model: fallback }),
        ],
      });

      // Assert
      await expect(out).rejects.toThrow();
      expect(fallback.doGenerate.mock.calls.length).toBe(0);
    });

    it('should invert with not()', async () => {
      // Arrange
      const primary = MockLanguageModel.from(retryableError);
      const fallback = MockLanguageModel.from(mockResultText);

      // Act
      const out = await retryableGenerateText({
        model: primary,
        prompt,
        retry: [not(httpStatus(503)).switch({ model: fallback })],
      });

      // Assert
      expect(out.text).toBe(mockResultText);
      expect(fallback.doGenerate.mock.calls.length).toBe(1);
    });
  });
});
