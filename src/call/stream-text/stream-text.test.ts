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

    describe('onSettled', () => {
      it('should fire at the commit point, before the stream is consumed', async () => {
        // Arrange — the boundary where this library's job ends. Firing later
        // would mean an operation whose stream nobody reads never settles.
        const model = MockLanguageModel.from({ doStream: mockStreamChunks });
        const onSettled = vi.fn();

        // Act
        const result = await retryableStreamText({
          model,
          prompt,
          retry: { retries: [], onSettled },
        });
        const firedBeforeConsuming = onSettled.mock.calls.length;
        await Streams.toArray(result.fullStream);

        // Assert
        expect(firedBeforeConsuming).toBe(1);
        expect(onSettled.mock.calls.length).toBe(1);
        expect(onSettled.mock.calls[0]![0].outcome).toBe('success');
      });

      it('should count every attempt when a retry rescued the stream', async () => {
        // Arrange
        const primary = MockLanguageModel.from({
          doStream: errorStreamChunks(retryableError),
        });
        const fallback = MockLanguageModel.from({ doStream: mockStreamChunks });
        const onSettled = vi.fn();

        // Act
        const result = await retryableStreamText({
          model: primary,
          prompt,
          retry: { retries: [fallback], onSettled },
        });
        await Streams.toArray(result.fullStream);

        // Assert
        const event = onSettled.mock.calls[0]![0];
        expect(event.outcome).toBe('success');
        expect(event.model).toBe(fallback);
        expect(event.attempts.length).toBe(2);
      });

      it('should report a failure when no attempt ever commits', async () => {
        // Arrange
        const model = MockLanguageModel.from({
          doStream: errorStreamChunks(retryableError),
        });
        const onSettled = vi.fn();

        // Act
        const result = retryableStreamText({
          model,
          prompt,
          retry: { retries: [], onSettled },
        });

        // Assert
        await expect(result).rejects.toThrow();
        const event = onSettled.mock.calls[0]![0];
        expect(event.outcome).toBe('failure');
        expect(event.attempts.length).toBe(1);
      });

      it('should report success for a stream that goes on to fail', async () => {
        // Arrange — the attempt committed, so the loop's job was done and its
        // recovery did work. What the stream does afterwards was never
        // recoverable, and `streamText`'s own `onError` reports it.
        const model = MockLanguageModel.from({
          doStream: [
            Language.streamStart(),
            ...Language.streamText('Par', { id: '1' }),
            Language.streamError(retryableError),
          ],
        });
        const onSettled = vi.fn();

        // Act
        const result = await retryableStreamText({
          model,
          prompt,
          onError: () => {},
          retry: { retries: [], onSettled },
        });
        const chunks = await Streams.toArray(result.fullStream);

        // Assert
        expect(onSettled.mock.calls[0]![0].outcome).toBe('success');
        expect(chunks.some((chunk) => chunk.type === 'error')).toBe(true);
      });

      it('should settle a stream that is never consumed', async () => {
        // Arrange — nothing here waits on the caller reading anything.
        const model = MockLanguageModel.from({ doStream: mockStreamChunks });
        const onSettled = vi.fn();

        // Act
        await retryableStreamText({
          model,
          prompt,
          retry: { retries: [], onSettled },
        });

        // Assert
        expect(onSettled.mock.calls.length).toBe(1);
      });
    });

    describe("the caller's own stream callbacks", () => {
      /**
       * Every attempt is issued with the caller's whole argument object, so
       * without holding them a discarded attempt reports a call that
       * completed, or an abort nobody saw, for output that was thrown away.
       */
      const record = (events: Array<string>) => ({
        onFinish: () => void events.push('onFinish'),
        onError: () => void events.push('onError'),
        onAbort: () => void events.push('onAbort'),
        onStepFinish: () => void events.push('onStepFinish'),
        onChunk: () => void events.push('onChunk'),
      });

      it('should stay silent for an attempt discarded on its result', async () => {
        // Arrange — the primary finishes with no content, so the loop reads it
        // to the end and fails over. That stream finished, and saying so would
        // be reporting a call the caller never received.
        const events: Array<string> = [];
        const primary = MockLanguageModel.from({
          doStream: contentFilterStreamChunks,
        });
        const fallback = MockLanguageModel.from({ doStream: mockStreamChunks });

        // Act
        const result = await retryableStreamText({
          model: primary,
          prompt,
          ...record(events),
          retry: [finishReason('content-filter').switch({ model: fallback })],
        });
        await Streams.toArray(result.fullStream);

        // Assert — one finish, belonging to the attempt the caller got.
        expect(events.filter((event) => event === 'onFinish').length).toBe(1);
      });

      it('should stay silent for an attempt discarded on a deadline', async () => {
        // Arrange — the primary opens its stream and stalls, so the deadline
        // aborts it. The caller never saw that abort.
        const events: Array<string> = [];
        const stalling = MockLanguageModel.from({
          doStream: async () => ({
            stream: new ReadableStream({
              start(controller) {
                controller.enqueue(Language.streamStart());
                /** Backstop, so an unaborted stream cannot hang the test. */
                setTimeout(() => {
                  try {
                    controller.close();
                  } catch {}
                }, 2_000).unref?.();
              },
            }),
          }),
        });
        const fallback = MockLanguageModel.from({ doStream: mockStreamChunks });

        // Act
        const result = await retryableStreamText({
          model: stalling,
          prompt,
          timeout: { firstChunkMs: 50 },
          ...record(events),
          retry: [fallback],
        });
        await Streams.toArray(result.fullStream);

        // Assert
        expect(events.includes('onAbort')).toBe(false);
        expect(events.filter((event) => event === 'onFinish').length).toBe(1);
      });

      it('should report the attempt that failed terminally', async () => {
        // Arrange — nothing recovered this one, so its error is the caller's.
        const events: Array<string> = [];
        const model = MockLanguageModel.from({
          doStream: errorStreamChunks(retryableError),
        });

        // Act
        const result = retryableStreamText({
          model,
          prompt,
          ...record(events),
          retry: {
            retries: [],
            onSettled: () => void events.push('onSettled'),
          },
        });

        // Assert — held back until the loop gave up, then released.
        await expect(result).rejects.toThrow();
        expect(events.includes('onError')).toBe(true);
        expect(events.indexOf('onSettled')).toBeLessThan(
          events.indexOf('onError'),
        );
      });

      it('should forward the winning attempt as its stream is consumed', async () => {
        // Arrange — nothing is held on the happy path beyond the settle: the
        // consumer drives these, and the loop settled long before.
        const events: Array<string> = [];
        const model = MockLanguageModel.from({ doStream: mockStreamChunks });

        // Act
        const result = await retryableStreamText({
          model,
          prompt,
          ...record(events),
          retry: [],
        });
        const beforeConsuming = [...events];
        await Streams.toArray(result.fullStream);

        // Assert
        expect(beforeConsuming.includes('onFinish')).toBe(false);
        expect(events.filter((event) => event === 'onFinish').length).toBe(1);
        expect(events.includes('onChunk')).toBe(true);
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

      it('should not report a recovered error to the caller', async () => {
        // Arrange — the primary fails before committing and the loop recovers,
        // so the caller receives the fallback's stream and never saw the
        // failure. `streamText`'s `onError` describes the call the caller got;
        // the retry's own `onError` is where recovered failures are reported.
        const callerOnError = vi.fn();
        const retryOnError = vi.fn();
        const primary = MockLanguageModel.from({
          doStream: errorStreamChunks(retryableError),
        });
        const fallback = MockLanguageModel.from({ doStream: mockStreamChunks });

        // Act
        const result = await retryableStreamText({
          model: primary,
          prompt,
          onError: callerOnError,
          retry: { retries: [fallback], onError: retryOnError },
        });
        await Streams.toArray(result.fullStream);

        // Assert
        expect(callerOnError.mock.calls.length).toBe(0);
        expect(retryOnError.mock.calls.length).toBe(1);
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
