import { streamText } from 'ai';
import { describe, expect, it } from 'vitest';
import { error } from '../../language-model/conditions/index.js';
import {
  createRetryableModel,
  Language,
  MockLanguageModel,
} from '../../internal/test-utils.js';
import type {
  LanguageModelStreamPart,
  LanguageModelStreamTransform,
} from '../../types.js';
import { RefusalError, refusalTransform } from './refusal-transform.js';

const REFUSAL = "I'm sorry, but I cannot assist with that request.";

/** Run a set of provider parts through a fresh transform and collect the output. */
const runTransform = async (
  transform: LanguageModelStreamTransform,
  input: Array<LanguageModelStreamPart>,
): Promise<Array<LanguageModelStreamPart>> => {
  const { readable, writable } = transform();
  const writer = writable.getWriter();
  const reader = readable.getReader();
  const output: Array<LanguageModelStreamPart> = [];
  const pump = (async () => {
    while (true) {
      const { done, value } = await reader.read();
      if (done) break;
      output.push(value);
    }
  })();
  for (const part of input) await writer.write(part);
  await writer.close();
  await pump;
  return output;
};

/** Concatenate the `delta` of every `text-delta` part. */
const textOf = (parts: Array<LanguageModelStreamPart>): string =>
  parts.map((p) => (p.type === 'text-delta' ? p.delta : '')).join('');

describe('refusalTransform', () => {
  describe('detection', () => {
    it('should emit an error part and drop the text when a refusal matches', async () => {
      // Arrange — a refusal split across deltas, finishing `stop`.
      const input = [
        Language.streamStart(),
        ...Language.streamText(
          [
            "I'm sorry, ",
            'but I cannot assist with that request.',
            ' Anything else?',
          ],
          {
            id: '1',
          },
        ),
        Language.streamFinish(),
      ];

      // Act
      const output = await runTransform(refusalTransform([REFUSAL]), input);

      // Assert — an error part carries a RefusalError, emitted before any text.
      // (Deltas after the match aren't withheld here; at the model layer the
      // reader cancels on the error, which cuts them off — see the integration
      // tests.)
      const errorIndex = output.findIndex((p) => p.type === 'error');
      expect(errorIndex).toBeGreaterThanOrEqual(0);
      expect((output[errorIndex] as { error?: unknown }).error).toBeInstanceOf(
        RefusalError,
      );
      const textBeforeError = textOf(output.slice(0, errorIndex));
      expect(textBeforeError).toBe('');
    });

    it('should forward a real answer untouched when the text diverges', async () => {
      // Arrange — "I'm sorry to hear" diverges from the refusal at "to" vs "but".
      const input = [
        Language.streamStart(),
        ...Language.streamText(["I'm sorry ", 'to hear that. Here is help.'], {
          id: '1',
        }),
        Language.streamFinish(),
      ];

      // Act
      const output = await runTransform(refusalTransform([REFUSAL]), input);

      // Assert
      expect(output.some((p) => p.type === 'error')).toBe(false);
      expect(textOf(output)).toBe("I'm sorry to hear that. Here is help.");
    });

    it('should flush held text when the stream ends on an inconclusive prefix', async () => {
      // Arrange — text stops while still a prefix of the phrase.
      const input = [
        Language.streamStart(),
        ...Language.streamText(["I'm sorry"], { id: '1' }),
        Language.streamFinish(),
      ];

      // Act
      const output = await runTransform(refusalTransform([REFUSAL]), input);

      // Assert — never resolved as a refusal, so the text is preserved.
      expect(output.some((p) => p.type === 'error')).toBe(false);
      expect(textOf(output)).toBe("I'm sorry");
    });

    it('should flush held text before a non-text part', async () => {
      // Arrange — a prefix delta, then a tool call arrives (non-text content).
      const input: Array<LanguageModelStreamPart> = [
        Language.streamStart(),
        { type: 'text-delta', id: '1', delta: "I'm sorry" },
        Language.toolCall({
          toolCallId: 't1',
          toolName: 'search',
          input: '{}',
        }),
      ];

      // Act
      const output = await runTransform(refusalTransform([REFUSAL]), input);

      // Assert — held text flushed ahead of the tool call, in order, no error.
      expect(output.some((p) => p.type === 'error')).toBe(false);
      const types = output.map((p) => p.type);
      expect(types.indexOf('text-delta')).toBeLessThan(
        types.indexOf('tool-call'),
      );
    });

    it('should match case- and whitespace-insensitively', async () => {
      // Arrange
      const input = [
        Language.streamStart(),
        ...Language.streamText(
          ["I'M   SORRY,\n BUT I CANNOT ASSIST WITH THAT REQUEST."],
          { id: '1' },
        ),
        Language.streamFinish(),
      ];

      // Act
      const output = await runTransform(refusalTransform([REFUSAL]), input);

      // Assert
      expect(output.some((p) => p.type === 'error')).toBe(true);
    });

    it('should emit a custom error from onRefusal', async () => {
      // Arrange
      const input = [
        Language.streamStart(),
        ...Language.streamText(
          ["I'm sorry, but I cannot assist with that request."],
          {
            id: '1',
          },
        ),
        Language.streamFinish(),
      ];
      const transform = refusalTransform([REFUSAL], {
        onRefusal: ({ phrase }) =>
          Object.assign(new Error(`blocked: ${phrase}`), {
            name: 'BlockedError',
          }),
      });

      // Act
      const output = await runTransform(transform, input);

      // Assert
      const errorPart = output.find((p) => p.type === 'error');
      expect((errorPart as { error?: Error }).error?.name).toBe('BlockedError');
    });
  });

  describe(`createRetryableModel integration`, () => {
    /** Provider parts for a stream of the given deltas, finishing `stop`. */
    const streaming = (deltas: Array<string>) =>
      MockLanguageModel.from({
        doStream: [
          Language.streamStart(),
          ...Language.streamText(deltas, { id: '1' }),
          Language.streamFinish(),
        ],
      });

    it('should recover a refusal at the model layer under plain streamText', async () => {
      // Arrange — the transform converts the refusal to an error the condition
      // matches, so the model layer fails over with no call-layer wrapper.
      const primary = streaming([
        "I'm sorry, ",
        'but I cannot assist with that request.',
      ]);
      const fallback = streaming(['clean answer']);
      const model = createRetryableModel({
        model: primary,
        retries: [
          error((e) => e instanceof RefusalError).switch({ model: fallback }),
        ],
        experimental_transform: refusalTransform([REFUSAL]),
      });

      // Act
      const result = streamText({ model, prompt: 'hi' });
      let text = '';
      for await (const delta of result.textStream) text += delta;

      // Assert
      expect(text).toBe('clean answer');
      expect(primary.doStream).toHaveBeenCalledTimes(1);
      expect(fallback.doStream).toHaveBeenCalledTimes(1);
    });

    it('should not fail over a real answer that shares a leading fragment', async () => {
      // Arrange
      const answer = "I'm sorry to hear that. Here is help.";
      const primary = streaming([answer]);
      const fallback = streaming(['clean answer']);
      const model = createRetryableModel({
        model: primary,
        retries: [
          error((e) => e instanceof RefusalError).switch({ model: fallback }),
        ],
        experimental_transform: refusalTransform([REFUSAL]),
      });

      // Act
      const result = streamText({ model, prompt: 'hi' });
      let text = '';
      for await (const delta of result.textStream) text += delta;

      // Assert
      expect(text).toBe(answer);
      expect(fallback.doStream).toHaveBeenCalledTimes(0);
    });

    it('should surface the refusal error when no condition matches', async () => {
      // Arrange — a condition that does NOT match RefusalError.
      const primary = streaming([
        "I'm sorry, but I cannot assist with that request.",
      ]);
      const fallback = streaming(['clean answer']);
      const errors: Array<Error> = [];
      const model = createRetryableModel({
        model: primary,
        retries: [
          error.message('some other error').switch({ model: fallback }),
        ],
        experimental_transform: refusalTransform([REFUSAL]),
      });

      // Act
      const result = streamText({
        model,
        prompt: 'hi',
        onError: ({ error }) => {
          errors.push(error as Error);
        },
      });
      let text = '';
      for await (const delta of result.textStream) text += delta;

      // Assert — refusal suppressed, error surfaced, fallback untouched.
      expect(text).toBe('');
      expect(errors.length).toBe(1);
      expect(errors[0]).toBeInstanceOf(RefusalError);
      expect(fallback.doStream).toHaveBeenCalledTimes(0);
    });
  });
});
