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
} from '../../internal/test-utils.js';
import {
  finishReason,
  result as resultCondition,
  timeout,
} from './conditions/index.js';
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

  describe('error-based retries', () => {
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

    describe('onCommit', () => {
      it('should fire before the stream is consumed, not after it ends', async () => {
        // Arrange — the boundary the hook is named for. If it fired at the end
        // of the stream instead, it could not have run before the first read.
        const model = MockLanguageModel.from({ doStream: mockStreamChunks });
        const onCommit = vi.fn();

        // Act
        const result = await retryableStreamText({
          model,
          prompt,
          retry: { retries: [], onCommit },
        });
        const firedBeforeConsuming = onCommit.mock.calls.length;
        await Streams.toArray(result.fullStream);

        // Assert
        expect(firedBeforeConsuming).toBe(1);
        expect(onCommit.mock.calls.length).toBe(1);
      });

      it('should report the attempt that committed and the ones retried before it', async () => {
        // Arrange
        const primary = MockLanguageModel.from({
          doStream: errorStreamChunks(retryableError),
        });
        const fallback = MockLanguageModel.from({ doStream: mockStreamChunks });
        const onCommit = vi.fn();

        // Act
        const result = await retryableStreamText({
          model: primary,
          prompt,
          retry: { retries: [fallback], onCommit },
        });
        await Streams.toArray(result.fullStream);

        // Assert
        const context = onCommit.mock.calls[0]![0];
        expect(context.current.model).toBe(fallback);
        expect(context.attempts.length).toBe(1);
        expect(context.current.type).toBe('success');
      });

      it('should fire even though the stream goes on to fail', async () => {
        // Arrange — the reason this is not called `onSuccess`. The attempt
        // commits at its first delta and the error arrives afterwards, in the
        // caller's stream, where no hook of ours can retract anything.
        const model = MockLanguageModel.from({
          doStream: [
            Language.streamStart(),
            ...Language.streamText('Par', { id: '1' }),
            Language.streamError(retryableError),
          ],
        });
        const onCommit = vi.fn();

        // Act
        const result = await retryableStreamText({
          model,
          prompt,
          retry: { retries: [], onCommit },
        });
        const chunks = await Streams.toArray(result.fullStream);

        // Assert
        expect(onCommit.mock.calls.length).toBe(1);
        expect(chunks.some((chunk) => chunk.type === 'error')).toBe(true);
      });

      it('should stay silent when no attempt ever commits', async () => {
        // Arrange
        const model = MockLanguageModel.from({
          doStream: errorStreamChunks(retryableError),
        });
        const onCommit = vi.fn();

        // Act
        const result = retryableStreamText({
          model,
          prompt,
          retry: { retries: [], onCommit },
        });

        // Assert
        await expect(result).rejects.toThrow();
        expect(onCommit.mock.calls.length).toBe(0);
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

    /**
     * `streamText` reports stream failures to its own `onError` rather than
     * throwing, so what the loop does with that handler is part of how it
     * recovers from an error.
     */
    describe('the SDK onError handler', () => {
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
            seen.push({ ...res });
            return true;
          }).switch({ model: fallback }),
        ],
      });
      await Streams.toArray(streamed.fullStream);

      // Assert — the whole of what a pre-commit stream can report. No content
      // field is among it, and that is a fact rather than an omission: any
      // content part would have committed the attempt beyond retry.
      expect(Object.keys(seen[0] as object).sort()).toEqual([
        'finishReason',
        'providerMetadata',
        'usage',
      ]);
      expect((seen[0] as { finishReason: string }).finishReason).toBe(
        'content-filter',
      );
    });
  });
});
