import { describe, expect, it, vi } from 'vitest';
import {
  chunksToText,
  contentFilterStreamChunks,
  errorStreamChunks,
  Language,
  MockLanguageModel,
  mockStreamChunks,
  retryableError,
  Streams,
} from '../../../internal/test-utils.js';
import {
  finishReason,
  isGenerateTextResult,
  isStreamTextResult,
  result as resultCondition,
  timeout,
} from '../conditions/index.js';
import { retryableStreamText } from './stream-text.js';

const prompt = 'Hello!';

describe('retryableStreamText', () => {
  describe('success', () => {
    it('should stream the first attempt when nothing fails', async () => {
      // Arrange
      const model = MockLanguageModel.from({ doStream: mockStreamChunks });

      // Act
      const result = await retryableStreamText({ model, prompt });
      const chunks = await Streams.toArray(result.fullStream);

      // Assert
      expect(chunksToText(chunks)).toBe('Hello, world!');
    });

    it('should leave the caller a full stream despite the commit read', async () => {
      // Arrange — commit detection reads the leading parts off a fresh tee, so
      // the consumer must still see them.
      const model = MockLanguageModel.from({ doStream: mockStreamChunks });

      // Act
      const result = await retryableStreamText({ model, prompt });
      const chunks = await Streams.toArray(result.fullStream);

      // Assert
      expect(chunks.some((c) => c.type === 'finish')).toBe(true);
      expect(chunksToText(chunks)).toBe('Hello, world!');
    });
  });

  describe('the commit boundary', () => {
    it('should fall over when the stream errors before any content', async () => {
      // Arrange
      const primary = MockLanguageModel.from({
        doStream: errorStreamChunks(retryableError),
      });
      const fallback = MockLanguageModel.from({ doStream: mockStreamChunks });

      // Act
      const result = await retryableStreamText({
        model: primary,
        prompt,
        retry: [fallback],
      });
      const chunks = await Streams.toArray(result.fullStream);

      // Assert
      expect(chunksToText(chunks)).toBe('Hello, world!');
    });

    it('should not fall over once content has been streamed', async () => {
      // Arrange — the attempt commits at its first text delta, so the error
      // that follows belongs to the caller's stream.
      const primary = MockLanguageModel.from({
        doStream: [
          Language.streamStart(),
          ...Language.streamText('Par', { id: '1' }),
          Language.streamError(retryableError),
        ],
      });
      const fallback = MockLanguageModel.from({ doStream: mockStreamChunks });

      // Act
      const result = await retryableStreamText({
        model: primary,
        prompt,
        retry: [fallback],
      });
      const chunks = await Streams.toArray(result.fullStream);

      // Assert
      expect(chunksToText(chunks)).toBe('Par');
      expect(fallback.doStream.mock.calls.length).toBe(0);
    });
  });

  describe('deadlines', () => {
    it('should fall over on a call-level timeout, which no model-level retry can see', async () => {
      // Arrange — this is the failure mode the call layer exists for: once the
      // deadline fires the SDK has torn the stream down, so a retry below the
      // model has nothing left to hand back.
      const stalling = MockLanguageModel.from({
        doStream: { chunks: mockStreamChunks, initialDelayInMs: 5_000 },
      });
      const fallback = MockLanguageModel.from({ doStream: mockStreamChunks });

      // Act
      const result = await retryableStreamText({
        model: stalling,
        prompt,
        timeout: { firstChunkMs: 50 },
        retry: [timeout().switch({ model: fallback })],
      });
      const chunks = await Streams.toArray(result.fullStream);

      // Assert
      expect(chunksToText(chunks)).toBe('Hello, world!');
    });

    it('should keep the finer-grained windows when a retry sets a total', async () => {
      // Arrange — `Retry.timeout` replaces `totalMs` only.
      const primary = MockLanguageModel.from({
        doStream: errorStreamChunks(retryableError),
      });
      const fallback = MockLanguageModel.from({ doStream: mockStreamChunks });

      // Act
      const result = await retryableStreamText({
        model: primary,
        prompt,
        timeout: { firstChunkMs: 5_000 },
        retry: [{ model: fallback, timeout: 5_000 }],
      });
      const chunks = await Streams.toArray(result.fullStream);

      // Assert
      expect(chunksToText(chunks)).toBe('Hello, world!');
    });
  });

  describe('result-based retries', () => {
    it('should fall over on a contentless content-filter finish', async () => {
      // Arrange — the stream finishes without ever emitting content, so the
      // attempt is still recoverable.
      const primary = MockLanguageModel.from({
        doStream: contentFilterStreamChunks,
      });
      const fallback = MockLanguageModel.from({ doStream: mockStreamChunks });

      // Act
      const result = await retryableStreamText({
        model: primary,
        prompt,
        retry: [finishReason('content-filter').switch({ model: fallback })],
      });
      const chunks = await Streams.toArray(result.fullStream);

      // Assert
      expect(chunksToText(chunks)).toBe('Hello, world!');
    });

    it('should report a stream judged before any content as a stream result', async () => {
      // Arrange — nothing was generated, so there is genuinely nothing to see,
      // and the reported result declares no content at all.
      const primary = MockLanguageModel.from({
        doStream: contentFilterStreamChunks,
      });
      const fallback = MockLanguageModel.from({ doStream: mockStreamChunks });
      const seen: Array<unknown> = [];

      // Act
      const streamed = await retryableStreamText({
        model: primary,
        prompt,
        retry: [
          resultCondition((res) => {
            seen.push({
              operation: res.operation,
              isStream: isStreamTextResult(res),
              isGenerate: isGenerateTextResult(res),
              finishReason: res.finishReason,
            });
            return true;
          }).switch({ model: fallback }),
        ],
      });
      await Streams.toArray(streamed.fullStream);

      // Assert
      expect(seen[0]).toEqual({
        operation: 'streamText',
        isStream: true,
        isGenerate: false,
        finishReason: 'content-filter',
      });
    });
  });

  describe('onError', () => {
    it('should not log recovered attempts through the SDK default handler', async () => {
      // Arrange — `streamText` defaults `onError` to `console.error`, which
      // would report every attempt the loop went on to recover.
      const consoleError = vi
        .spyOn(console, 'error')
        .mockImplementation(() => {});
      const primary = MockLanguageModel.from({
        doStream: errorStreamChunks(retryableError),
      });
      const fallback = MockLanguageModel.from({ doStream: mockStreamChunks });

      // Act
      const result = await retryableStreamText({
        model: primary,
        prompt,
        retry: [fallback],
      });
      await Streams.toArray(result.fullStream);

      // Assert
      expect(consoleError.mock.calls.length).toBe(0);
      consoleError.mockRestore();
    });

    it('should keep a caller-supplied onError', async () => {
      // Arrange
      const onError = vi.fn();
      const primary = MockLanguageModel.from({
        doStream: errorStreamChunks(retryableError),
      });
      const fallback = MockLanguageModel.from({ doStream: mockStreamChunks });

      // Act
      const result = await retryableStreamText({
        model: primary,
        prompt,
        onError,
        retry: [fallback],
      });
      await Streams.toArray(result.fullStream);

      // Assert
      expect(onError.mock.calls.length).toBe(1);
    });
  });
});
