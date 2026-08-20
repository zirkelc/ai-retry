import { generateText, tool } from 'ai';
import { describe, expect, it } from 'vitest';
import { z } from 'zod';
import {
  MockLanguageModel,
  mockResult,
  mockResultText,
  nonRetryableError,
  retryableError,
} from '../../internal/test-utils.js';
import type { LanguageModel } from '../../types.js';
import { finishReason, result as resultCondition } from './conditions/index.js';
import { retryableGenerateText } from './generate-text.js';

const prompt = 'Hello!';

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
      // Arrange — the loop passes the result straight through, so what the
      // caller gets is the SDK's own object rather than a copy or a wrapper.
      const model = MockLanguageModel.from(mockResultText);

      // Act
      const direct = await generateText({ model, prompt });
      const wrapped = await retryableGenerateText({ model, prompt });

      // Assert — the SDK exposes most of a result through prototype getters, so
      // sameness of prototype is what says nothing rebuilt it on the way out.
      expect(Object.getPrototypeOf(wrapped)).toBe(
        Object.getPrototypeOf(direct),
      );
      expect(wrapped.text).toBe(mockResultText);
    });
  });

  describe('error-based retries', () => {
    it('should fall over to the next model after an error', async () => {
      // Arrange
      const primary = MockLanguageModel.from(retryableError);
      const fallback = MockLanguageModel.from(mockResultText);

      // Act
      const result = await retryableGenerateText({
        model: primary,
        prompt,
        retry: [fallback],
      });

      // Assert
      expect(result.text).toBe(mockResultText);
      expect(fallback.doGenerate.mock.calls.length).toBe(1);
    });

    it('should surface the error when no retry matched', async () => {
      // Arrange
      const primary = MockLanguageModel.from(nonRetryableError);

      // Act
      const result = retryableGenerateText({
        model: primary,
        prompt,
        retry: [],
      });

      // Assert
      await expect(result).rejects.toThrow(nonRetryableError);
    });

    it('should override the prompt for the retry attempt', async () => {
      // Arrange
      const primary = MockLanguageModel.from(retryableError);
      const fallback = MockLanguageModel.from(mockResultText);

      // Act
      await retryableGenerateText({
        model: primary,
        prompt,
        retry: [{ model: fallback, options: { prompt: 'Rephrased!' } }],
      });

      // Assert — the override reaches the model as a real prompt, not a
      // provider-shaped message array.
      expect(fallback.doGenerate.mock.calls[0]![0].prompt).toEqual([
        { role: 'user', content: [{ type: 'text', text: 'Rephrased!' }] },
      ]);
    });

    describe('deadlines', () => {
      it('should apply a retry timeout through the timeout argument', async () => {
        // Arrange — `generateText` has a `timeout` of its own, so the deadline
        // goes there rather than into `abortSignal`. The retry gets 50ms against
        // a model that needs 5s.
        const primary = MockLanguageModel.from(retryableError);
        const slow = MockLanguageModel.from({
          doGenerate: { content: [], delayInMs: 5_000 },
        });
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
      // `text` is the SDK's flat field; a provider result would carry `content`.
      const primary = MockLanguageModel.from('Too short.');
      const fallback = MockLanguageModel.from(mockResultText);
      const seen: Array<string> = [];

      // Act
      const result = await retryableGenerateText({
        model: primary,
        prompt,
        retry: [
          resultCondition((res) => {
            seen.push(res.text);
            return res.text === 'Too short.';
          }).switch({ model: fallback }),
        ],
      });

      // Assert — the predicate saw what the model actually generated.
      expect(seen[0]).toBe('Too short.');
      expect(result.text).toBe(mockResultText);
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
      const tools = {
        lookup: tool({
          description: 'look a city up',
          inputSchema: z.object({ city: z.string() }),
        }),
      };

      // Act — the tool set is named at the condition, which is the only place
      // it can be: a condition is written against the entry point, not against
      // one call site, so there is nothing to infer it from.
      await retryableGenerateText({
        model: primary,
        prompt,
        tools,
        retry: [
          resultCondition<typeof tools>((res) => {
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
