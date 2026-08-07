import { describe, expect, it, vi } from 'vitest';
import { detectStreamCommit } from './detect-stream-commit.js';

/**
 * The parts `streamText` puts on its `stream` getter, which is what commit
 * detection reads. Built directly rather than through the SDK so each terminal
 * case can be produced in isolation.
 */
const streamOf = (
  parts: Array<unknown>,
  onCancel?: () => void,
): ReadableStream<unknown> =>
  new ReadableStream({
    start(controller) {
      for (const part of parts) controller.enqueue(part);
      controller.close();
    },
    cancel: onCancel,
  });

/** A stream that never closes, so only an early return can end the read. */
const endlessStreamOf = (
  parts: Array<unknown>,
  onCancel?: () => void,
): ReadableStream<unknown> =>
  new ReadableStream({
    start(controller) {
      for (const part of parts) controller.enqueue(part);
    },
    cancel: onCancel,
  });

const finishPart = (finishReason = 'stop') => ({
  type: 'finish',
  finishReason,
  totalUsage: {
    inputTokens: 10,
    outputTokens: 0,
    totalTokens: 10,
    inputTokenDetails: {
      noCacheTokens: 10,
      cacheReadTokens: 0,
      cacheWriteTokens: 0,
    },
    outputTokenDetails: { textTokens: 0, reasoningTokens: 0 },
  },
});

describe('detectStreamCommit', () => {
  describe('commit', () => {
    it('should commit on the first content part', async () => {
      // Arrange
      const stream = endlessStreamOf([
        { type: 'start' },
        { type: 'text-start', id: '1' },
        { type: 'text-delta', id: '1', delta: 'hi' },
      ]);

      // Act
      const settled = await detectStreamCommit(stream, undefined);

      // Assert
      expect(settled.type).toBe('committed');
    });

    it('should commit on a tool call, not only on text', async () => {
      // Arrange
      const stream = endlessStreamOf([
        { type: 'start' },
        { type: 'tool-call', toolCallId: '1', toolName: 'x', input: {} },
      ]);

      // Act
      const settled = await detectStreamCommit(stream, undefined);

      // Assert
      expect(settled.type).toBe('committed');
    });

    it('should commit when the stream ends without a finish part', async () => {
      // Arrange — nothing to judge, so the attempt is the caller's.
      const stream = streamOf([{ type: 'start' }]);

      // Act
      const settled = await detectStreamCommit(stream, undefined);

      // Assert
      expect(settled.type).toBe('committed');
    });

    it('should not treat preamble parts as content', async () => {
      // Arrange
      const stream = streamOf([
        { type: 'start' },
        { type: 'start-step' },
        { type: 'response-metadata', id: 'id-0' },
        finishPart('content-filter'),
      ]);

      // Act
      const settled = await detectStreamCommit(stream, undefined);

      // Assert — reached the finish, so it is still judgeable.
      expect(settled.type).toBe('result');
    });
  });

  describe('contentless finish', () => {
    it('should report the finish reason and usage', async () => {
      // Arrange
      const stream = streamOf([
        { type: 'start' },
        finishPart('content-filter'),
      ]);

      // Act
      const settled = await detectStreamCommit(stream, undefined);

      // Assert
      expect(settled.type).toBe('result');
      if (settled.type !== 'result') return;
      expect(settled.result.operation).toBe('streamText');
      expect(settled.result.finishReason).toBe('content-filter');
      expect(settled.result.usage.inputTokens).toBe(10);
    });

    it('should carry provider metadata from the finish-step part', async () => {
      // Arrange
      const stream = streamOf([
        { type: 'start' },
        {
          type: 'finish-step',
          providerMetadata: { openai: { flagged: true } },
        },
        finishPart('content-filter'),
      ]);

      // Act
      const settled = await detectStreamCommit(stream, undefined);

      // Assert
      expect(settled.type).toBe('result');
      if (settled.type !== 'result') return;
      expect(settled.result.providerMetadata).toEqual({
        openai: { flagged: true },
      });
    });

    it('should keep earlier provider metadata when a later step carries none', async () => {
      // Arrange — only some steps report metadata, and a later bare one must
      // not erase what an earlier one established.
      const stream = streamOf([
        { type: 'start' },
        {
          type: 'finish-step',
          providerMetadata: { openai: { flagged: true } },
        },
        { type: 'finish-step' },
        finishPart('content-filter'),
      ]);

      // Act
      const settled = await detectStreamCommit(stream, undefined);

      // Assert
      expect(settled.type).toBe('result');
      if (settled.type !== 'result') return;
      expect(settled.result.providerMetadata).toEqual({
        openai: { flagged: true },
      });
    });

    it('should declare no content, since there is none by construction', async () => {
      // Arrange
      const stream = streamOf([{ type: 'start' }, finishPart()]);

      // Act
      const settled = await detectStreamCommit(stream, undefined);

      // Assert
      expect(settled.type).toBe('result');
      if (settled.type !== 'result') return;
      expect('text' in settled.result).toBe(false);
      expect('content' in settled.result).toBe(false);
    });
  });

  describe('failures', () => {
    it('should throw the error carried by an error part', async () => {
      // Arrange
      const boom = new Error('boom');
      const stream = streamOf([
        { type: 'start' },
        { type: 'error', error: boom },
      ]);

      // Act
      const settled = detectStreamCommit(stream, undefined);

      // Assert
      await expect(settled).rejects.toThrow();
      await settled.catch((e) => expect(e).toBe(boom));
    });

    it('should reconstruct a named error from an abort part', async () => {
      // Arrange — a call-level deadline aborts an internal controller, so only
      // the serialized reason survives.
      const stream = streamOf([
        { type: 'start' },
        { type: 'abort', reason: 'TimeoutError: signal timed out' },
      ]);

      // Act
      const settled = detectStreamCommit(stream, undefined);

      // Assert — the name is restored so `timeout()` still matches.
      await settled.catch((e) => {
        expect((e as Error).name).toBe('TimeoutError');
        expect((e as Error).message).toBe('signal timed out');
      });
      await expect(settled).rejects.toThrow();
    });

    it('should fall back to a plain error for an unnamed abort reason', async () => {
      // Arrange
      const stream = streamOf([
        { type: 'start' },
        { type: 'abort', reason: 'just stopped' },
      ]);

      // Act
      const settled = detectStreamCommit(stream, undefined);

      // Assert
      await settled.catch((e) => {
        expect((e as Error).name).toBe('Error');
        expect((e as Error).message).toBe('just stopped');
      });
      await expect(settled).rejects.toThrow();
    });

    it('should fall back to a generic error when the abort carries no reason', async () => {
      // Arrange
      const stream = streamOf([{ type: 'start' }, { type: 'abort' }]);

      // Act
      const settled = detectStreamCommit(stream, undefined);

      // Assert
      await settled.catch((e) =>
        expect((e as Error).message).toBe('stream aborted'),
      );
      await expect(settled).rejects.toThrow();
    });

    it("should prefer the caller's own abort reason when their signal fired", async () => {
      // Arrange — a genuine cancel has a structured reason, so `instanceof`
      // checks survive.
      const reason = new Error('caller cancelled');
      const controller = new AbortController();
      controller.abort(reason);
      const stream = streamOf([
        { type: 'start' },
        { type: 'abort', reason: 'TimeoutError: signal timed out' },
      ]);

      // Act
      const settled = detectStreamCommit(stream, controller.signal);

      // Assert
      await settled.catch((e) => expect(e).toBe(reason));
      await expect(settled).rejects.toThrow();
    });
  });

  describe('the reader', () => {
    it('should cancel once the outcome is known', async () => {
      // Arrange
      const onCancel = vi.fn();
      const stream = endlessStreamOf(
        [{ type: 'text-delta', id: '1', delta: 'hi' }],
        onCancel,
      );

      // Act
      await detectStreamCommit(stream, undefined);
      await vi.waitFor(() => expect(onCancel.mock.calls.length).toBe(1));

      // Assert
      expect(onCancel.mock.calls.length).toBe(1);
    });

    it('should cancel when the attempt failed', async () => {
      // Arrange
      const onCancel = vi.fn();
      const stream = endlessStreamOf(
        [{ type: 'error', error: new Error('boom') }],
        onCancel,
      );

      // Act
      await detectStreamCommit(stream, undefined).catch(() => {});
      await vi.waitFor(() => expect(onCancel.mock.calls.length).toBe(1));

      // Assert
      expect(onCancel.mock.calls.length).toBe(1);
    });
  });
});
