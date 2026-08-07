import { tool } from 'ai';
import { describe, expect, it } from 'vitest';
import { z } from 'zod';
import {
  MockLanguageModel,
  mockResult,
  mockResultText,
  retryableError,
} from '../../../internal/test-utils.js';
import type { LanguageModel } from '../../../types.js';
import {
  finishReason,
  isGenerateTextResult,
  result as resultCondition,
} from '../conditions/index.js';
import { retryableGenerateText } from './generate-text.js';

const prompt = 'Hello!';

/** A `doGenerate` that takes `ms` to answer, and rejects if aborted first. */
const slowGenerate =
  (ms: number): LanguageModel['doGenerate'] =>
  async ({ abortSignal }) => {
    await new Promise<void>((resolve, reject) => {
      const handle = setTimeout(resolve, ms);
      abortSignal?.addEventListener('abort', () => {
        clearTimeout(handle);
        reject(abortSignal.reason);
      });
    });
    return mockResult;
  };

describe('retryableGenerateText', () => {
  describe('success', () => {
    it('should return the first attempt when nothing fails', async () => {
      // Arrange
      const model = MockLanguageModel.from(mockResultText);

      // Act
      const result = await retryableGenerateText({ model, prompt });

      // Assert
      expect(result.text).toBe(mockResultText);
      expect(model.doGenerate.mock.calls.length).toBe(1);
    });

    it('should hand back the SDK result untouched', async () => {
      // Arrange — the loop tags a view for conditions to read; the caller must
      // still receive the entry point's own object.
      const model = MockLanguageModel.from(mockResultText);

      // Act
      const result = await retryableGenerateText({ model, prompt });

      // Assert
      expect('operation' in result).toBe(false);
      expect(result.finishReason).toBe('stop');
      expect(result.usage.outputTokens).toBeGreaterThan(0);
    });
  });

  describe('deadlines', () => {
    it('should apply a retry timeout through the timeout argument', async () => {
      // Arrange — `generateText` has a `timeout` of its own, so the deadline
      // goes there rather than into `abortSignal`. The retry gets 50ms against
      // a model that needs 5s.
      const primary = MockLanguageModel.from(retryableError);
      const slow = MockLanguageModel.from({ doGenerate: slowGenerate(5_000) });
      const rescue = MockLanguageModel.from(mockResultText);

      // Act
      const result = await retryableGenerateText({
        model: primary,
        prompt,
        retry: [{ model: slow, timeout: 50 }, rescue],
      });

      // Assert — the deadline fired and the third model answered.
      expect(result.text).toBe(mockResultText);
      expect(rescue.doGenerate.mock.calls.length).toBe(1);
    });

    it('should leave the caller signal alone when no retry sets a timeout', async () => {
      // Arrange
      const model = MockLanguageModel.from(mockResultText);

      // Act
      await retryableGenerateText({ model, prompt });

      // Assert
      expect(model.doGenerate.mock.calls[0]![0].abortSignal).toBeUndefined();
    });
  });

  describe('result-based retries', () => {
    it('should fall over on a content-filter finish reason', async () => {
      // Arrange
      const primary = MockLanguageModel.from({
        content: [],
        finishReason: 'content-filter',
      });
      const fallback = MockLanguageModel.from(mockResultText);

      // Act
      const result = await retryableGenerateText({
        model: primary,
        prompt,
        retry: [finishReason('content-filter').switch({ model: fallback })],
      });

      // Assert
      expect(result.text).toBe(mockResultText);
      expect(fallback.doGenerate.mock.calls.length).toBe(1);
    });

    it('should give a result condition the generated text', async () => {
      // Arrange — a condition that reads the text, not just the finish reason.
      const primary = MockLanguageModel.from('Too short.');
      const fallback = MockLanguageModel.from(mockResultText);
      const seen: Array<string> = [];

      // Act
      const result = await retryableGenerateText({
        model: primary,
        prompt,
        retry: [
          resultCondition((res) => {
            if (!isGenerateTextResult(res)) return false;
            seen.push(res.text);
            return res.text === 'Too short.';
          }).switch({ model: fallback }),
        ],
      });

      // Assert — the predicate saw what the model actually generated.
      expect(seen[0]).toBe('Too short.');
      expect(result.text).toBe(mockResultText);
    });

    it('should report the operation that produced the result', async () => {
      // Arrange
      const primary = MockLanguageModel.from('Too short.');
      const fallback = MockLanguageModel.from(mockResultText);
      const seen: Array<string> = [];

      // Act
      await retryableGenerateText({
        model: primary,
        prompt,
        retry: [
          resultCondition((res) => {
            seen.push(res.operation);
            return true;
          }).switch({ model: fallback }),
        ],
      });

      // Assert
      expect(seen[0]).toBe('generateText');
    });

    it('should give a result condition the tool calls that were made', async () => {
      // Arrange — the entry point's own result, so the input arrives parsed.
      const primary = MockLanguageModel.from({
        content: [
          {
            type: 'tool-call',
            toolCallId: '1',
            toolName: 'lookup',
            input: '{"city":"Berlin"}',
          },
        ],
        finishReason: 'tool-calls',
      });
      const fallback = MockLanguageModel.from(mockResultText);
      const seen: Array<unknown> = [];

      // Act
      await retryableGenerateText({
        model: primary,
        prompt,
        tools: {
          lookup: tool({
            description: 'look a city up',
            inputSchema: z.object({ city: z.string() }),
          }),
        },
        retry: [
          resultCondition((res) => {
            if (!isGenerateTextResult(res)) return false;
            seen.push(...res.toolCalls);
            return true;
          }).switch({ model: fallback }),
        ],
      });

      // Assert
      expect(seen.length).toBe(1);
      expect(seen[0]).toMatchObject({
        type: 'tool-call',
        toolCallId: '1',
        toolName: 'lookup',
        input: { city: 'Berlin' },
      });
    });

    it('should return the result when no result condition matches', async () => {
      // Arrange
      const primary = MockLanguageModel.from({
        content: [],
        finishReason: 'length',
      });
      const fallback = MockLanguageModel.from(mockResultText);

      // Act
      const result = await retryableGenerateText({
        model: primary,
        prompt,
        retry: [finishReason('content-filter').switch({ model: fallback })],
      });

      // Assert
      expect(result.finishReason).toBe('length');
      expect(fallback.doGenerate.mock.calls.length).toBe(0);
    });
  });
});
