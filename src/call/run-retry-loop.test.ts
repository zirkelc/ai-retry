import { RetryError } from 'ai';
import { describe, expect, it, vi } from 'vitest';
import {
  attemptSpans,
  createSpanExporter,
  Embedding,
  findSpan,
  MockEmbeddingModel,
  MockImageModel,
  mockImageResult,
  MockLanguageModel,
  mockResultText,
  mockStreamChunks,
  nonRetryableError,
  retryableError,
  Streams,
} from '../internal/test-utils.js';
import { retryableEmbed } from './embed/embed.js';
import { retryableEmbedMany } from './embed-many/embed-many.js';
import { retryableGenerateImage } from './generate-image/generate-image.js';
import { aborted, finishReason } from './generate-text/conditions/index.js';
import { retryableGenerateText } from './generate-text/generate-text.js';
import { retryableStreamText } from './stream-text/stream-text.js';

/**
 * The retry loop is shared by every call-level entry point, so its behavior is
 * exercised once here rather than repeated five times. `retryableGenerateText`
 * stands in as the representative caller; what is asserted is the loop's, not
 * `generateText`'s — anything specific to an entry point (its deadline
 * strategy, how its outcome is decided, what it returns) lives beside that
 * entry point instead.
 */

const prompt = 'Hello!';

describe('the SDK in-call retries', () => {
  it('should be disabled by default', async () => {
    // Arrange — left at the SDK default the entry point would re-issue the
    // failing model before the loop ever saw the error.
    const model = MockLanguageModel.from(retryableError);

    // Act
    const result = retryableGenerateText({ model, prompt });

    // Assert
    await expect(result).rejects.toThrow();
    expect(model.doGenerate.mock.calls.length).toBe(1);
  });

  it('should be kept when the caller sets maxRetries explicitly', async () => {
    // Arrange
    const model = MockLanguageModel.from(retryableError);

    // Act
    const result = retryableGenerateText({ model, prompt, maxRetries: 1 });

    // Assert — one original call plus one SDK-level retry.
    await expect(result).rejects.toThrow();
    expect(model.doGenerate.mock.calls.length).toBe(2);
  });
});

describe('fail-over', () => {
  it('should move to the next model after an error', async () => {
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
    expect(primary.doGenerate.mock.calls.length).toBe(1);
    expect(fallback.doGenerate.mock.calls.length).toBe(1);
  });

  it('should throw a RetryError once more than one attempt was made', async () => {
    // Arrange
    const primary = MockLanguageModel.from(retryableError);
    const fallback = MockLanguageModel.from(nonRetryableError);

    // Act
    const result = retryableGenerateText({
      model: primary,
      prompt,
      retry: [fallback],
    });

    // Assert
    await expect(result).rejects.toThrow(RetryError);
  });

  it('should throw the original error when no retry matched', async () => {
    // Arrange
    const primary = MockLanguageModel.from(nonRetryableError);

    // Act
    const result = retryableGenerateText({
      model: primary,
      prompt,
      retry: [aborted().switch({ model: MockLanguageModel.from() })],
    });

    // Assert — a single attempt surfaces its own error, unwrapped.
    await expect(result).rejects.toThrow(nonRetryableError);
  });

  it('should apply a backoff delay before retrying', async () => {
    // Arrange
    const primary = MockLanguageModel.from(retryableError);
    const fallback = MockLanguageModel.from(mockResultText);

    // Act
    const start = Date.now();
    await retryableGenerateText({
      model: primary,
      prompt,
      retry: [{ model: fallback, delay: 120 }],
    });
    const elapsed = Date.now() - start;

    // Assert
    expect(elapsed).toBeGreaterThanOrEqual(100);
  });

  it('should apply a backoff delay before a result-based retry too', async () => {
    // Arrange — the delay is honored whichever branch decided to retry, not
    // only the error one.
    const primary = MockLanguageModel.from({
      content: [],
      finishReason: 'content-filter',
    });
    const fallback = MockLanguageModel.from(mockResultText);

    // Act
    const start = Date.now();
    await retryableGenerateText({
      model: primary,
      prompt,
      retry: [
        finishReason('content-filter').switch({ model: fallback, delay: 120 }),
      ],
    });
    const elapsed = Date.now() - start;

    // Assert
    expect(elapsed).toBeGreaterThanOrEqual(100);
    expect(fallback.doGenerate.mock.calls.length).toBe(1);
  });

  it('should not fail over once the caller has cancelled', async () => {
    // Arrange
    const controller = new AbortController();
    const primary = MockLanguageModel.from({
      doGenerate: async () => {
        controller.abort();
        throw retryableError;
      },
    });
    const fallback = MockLanguageModel.from(mockResultText);

    // Act
    const result = retryableGenerateText({
      model: primary,
      prompt,
      abortSignal: controller.signal,
      retry: [fallback],
    });

    // Assert — a re-run would forward the dead signal and abort instantly.
    await expect(result).rejects.toThrow();
    expect(fallback.doGenerate.mock.calls.length).toBe(0);
  });
});

describe('argument overrides', () => {
  it('should apply Retry.options to the retry attempt', async () => {
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
    const callOptions = fallback.doGenerate.mock.calls[0]![0];
    expect(callOptions.prompt).toEqual([
      { role: 'user', content: [{ type: 'text', text: 'Rephrased!' }] },
    ]);
  });

  it('should let onRetry outrank Retry.options per field', async () => {
    // Arrange
    const primary = MockLanguageModel.from(retryableError);
    const fallback = MockLanguageModel.from(mockResultText);

    // Act
    await retryableGenerateText({
      model: primary,
      prompt,
      retry: {
        retries: [
          {
            model: fallback,
            options: { prompt: 'From the retry', temperature: 0.1 },
          },
        ],
        onRetry: () => ({ options: { prompt: 'From onRetry' } }),
      },
    });

    // Assert — the hook wins on `prompt`, the retry keeps `temperature`.
    const callOptions = fallback.doGenerate.mock.calls[0]![0];
    expect(callOptions.prompt).toEqual([
      { role: 'user', content: [{ type: 'text', text: 'From onRetry' }] },
    ]);
    expect(callOptions.temperature).toBe(0.1);
  });
});

describe('hooks', () => {
  it('should report every failed attempt to onError', async () => {
    // Arrange
    const primary = MockLanguageModel.from(retryableError);
    const fallback = MockLanguageModel.from(mockResultText);
    const onError = vi.fn();

    // Act
    await retryableGenerateText({
      model: primary,
      prompt,
      retry: { retries: [fallback], onError },
    });

    // Assert
    expect(onError.mock.calls.length).toBe(1);
    expect(onError.mock.calls[0]![0].current.error).toBe(retryableError);
  });

  it('should report the entry point arguments on the failed attempt', async () => {
    // Arrange — the call's own arguments, not provider call options.
    const primary = MockLanguageModel.from(retryableError);
    const fallback = MockLanguageModel.from(mockResultText);
    const seen: Array<unknown> = [];

    // Act
    await retryableGenerateText({
      model: primary,
      prompt,
      temperature: 0.3,
      retry: {
        retries: [fallback],
        onError: (context) => {
          seen.push({
            prompt: context.current.options.prompt,
            temperature: context.current.options.temperature,
          });
        },
      },
    });

    // Assert
    expect(seen[0]).toEqual({ prompt, temperature: 0.3 });
  });

  describe('onSettled', () => {
    /**
     * The four outcomes the hook exists to tell apart, since the question it
     * answers is how often retrying rescues a call. `attempts.length` counts
     * every attempt, terminal one included, so it reads the same on both
     * paths: 1 means no retry happened, more means one did.
     */
    it('should report a success that took no retry', async () => {
      // Arrange
      const model = MockLanguageModel.from(mockResultText);
      const onSettled = vi.fn();

      // Act
      await retryableGenerateText({
        model,
        prompt,
        retry: { retries: [], onSettled },
      });

      // Assert
      const event = onSettled.mock.calls[0]![0];
      expect(event.outcome).toBe('success');
      expect(event.attempts.length).toBe(1);
      expect(event.model).toBe(model);
      expect(event.result.text).toBe(mockResultText);
    });

    it('should report a success that a retry rescued', async () => {
      // Arrange
      const primary = MockLanguageModel.from(retryableError);
      const fallback = MockLanguageModel.from(mockResultText);
      const onSettled = vi.fn();

      // Act
      await retryableGenerateText({
        model: primary,
        prompt,
        retry: { retries: [fallback], onSettled },
      });

      // Assert
      const event = onSettled.mock.calls[0]![0];
      expect(event.outcome).toBe('success');
      expect(event.attempts.length).toBe(2);
      expect(event.model).toBe(fallback);
      expect(event.attempts.at(-1).type).toBe('success');
      expect(event.attempts[0].type).toBe('error');
    });

    it('should report a failure that took no retry', async () => {
      // Arrange
      const model = MockLanguageModel.from(nonRetryableError);
      const onSettled = vi.fn();

      // Act
      const result = retryableGenerateText({
        model,
        prompt,
        retry: { retries: [], onSettled },
      });

      // Assert
      await expect(result).rejects.toThrow();
      const event = onSettled.mock.calls[0]![0];
      expect(event.outcome).toBe('failure');
      expect(event.attempts.length).toBe(1);
      expect(event.error).toBe(nonRetryableError);
    });

    it('should report a failure that a retry could not rescue', async () => {
      // Arrange
      const primary = MockLanguageModel.from(retryableError);
      const fallback = MockLanguageModel.from(retryableError);
      const onSettled = vi.fn();

      // Act
      const result = retryableGenerateText({
        model: primary,
        prompt,
        retry: { retries: [fallback], onSettled },
      });

      // Assert
      await expect(result).rejects.toThrow();
      const event = onSettled.mock.calls[0]![0];
      expect(event.outcome).toBe('failure');
      expect(event.attempts.length).toBe(2);
    });

    it('should fire exactly once, whichever way the call goes', async () => {
      // Arrange
      const onSettled = vi.fn();

      // Act
      await retryableGenerateText({
        model: MockLanguageModel.from(mockResultText),
        prompt,
        retry: { retries: [], onSettled },
      });
      await retryableGenerateText({
        model: MockLanguageModel.from(nonRetryableError),
        prompt,
        retry: { retries: [], onSettled },
      }).catch(() => {});

      // Assert
      expect(onSettled.mock.calls.length).toBe(2);
    });
  });
});

describe('disabled', () => {
  it('should behave like a direct call', async () => {
    // Arrange
    const primary = MockLanguageModel.from(nonRetryableError);
    const fallback = MockLanguageModel.from(mockResultText);
    const onSettled = vi.fn();

    // Act
    const result = retryableGenerateText({
      model: primary,
      prompt,
      maxRetries: 0,
      retry: { retries: [fallback], disabled: true, onSettled },
    });

    // Assert
    await expect(result).rejects.toThrow();
    expect(fallback.doGenerate.mock.calls.length).toBe(0);
    expect(onSettled.mock.calls.length).toBe(0);
  });
});

describe('telemetry', () => {
  it('should emit an operation span with one span per attempt', async () => {
    // Arrange
    const { exporter, tracer } = createSpanExporter();
    const primary = MockLanguageModel.from(retryableError);
    const fallback = MockLanguageModel.from(mockResultText);

    // Act
    await retryableGenerateText({
      model: primary,
      prompt,
      retry: {
        retries: [fallback],
        telemetry: { isEnabled: true, tracer },
      },
    });

    // Assert
    const operation = findSpan(exporter, 'ai_retry.generateText');
    expect(operation.attributes['ai_retry.outcome']).toBe('success');
    expect(operation.attributes['ai_retry.attempts']).toBe(2);

    const attempts = attemptSpans(exporter);
    expect(attempts.length).toBe(2);
    expect(attempts[0]!.attributes['ai_retry.attempt.outcome']).toBe('retry');
    expect(attempts[1]!.attributes['ai_retry.attempt.outcome']).toBe('success');
  });

  it('should make the attempt the active span, so callees nest under it', async () => {
    // Arrange — the entry point and the provider open their own spans from
    // whatever context is active when they run. Unless the attempt is active
    // by then they attach to whatever surrounded the retryable call, and the
    // model's work is rendered beside the retry tree rather than inside it.
    // This mock opens a span exactly as the SDK does.
    const { exporter, tracer } = createSpanExporter();
    const { context } = await import('@opentelemetry/api');
    const parentIdOf = (span: unknown): string | undefined =>
      (
        span as {
          parentSpanContext?: { spanId: string };
          parentSpanId?: string;
        }
      ).parentSpanContext?.spanId ??
      (span as { parentSpanId?: string }).parentSpanId;

    const opensASpan = MockLanguageModel.from(mockResultText);
    const issued = opensASpan.doGenerate;
    opensASpan.doGenerate = vi.fn(async (options: never) => {
      tracer.startSpan('callee', undefined, context.active()).end();
      return issued(options);
    }) as typeof opensASpan.doGenerate;

    // Act
    await retryableGenerateText({
      model: MockLanguageModel.from(retryableError),
      prompt,
      retry: {
        retries: [opensASpan],
        telemetry: { isEnabled: true, tracer },
      },
    });

    // Assert
    const callee = findSpan(exporter, 'callee');
    const attempts = attemptSpans(exporter);
    expect(parentIdOf(callee)).toBe(attempts.at(-1)!.spanContext().spanId);
  });

  /**
   * The span name and the standard `gen_ai.operation.name` are the one piece of
   * telemetry each entry point supplies itself, so the claim is made for all
   * five here rather than beside whichever one happened to have a test.
   */
  it.each([
    {
      operation: 'generateText',
      genAiOperation: 'chat',
      call: (args: any) => retryableGenerateText(args),
      model: () => MockLanguageModel.from(mockResultText),
      args: { prompt },
    },
    {
      operation: 'streamText',
      genAiOperation: 'chat',
      call: async (args: any) => {
        const out = await retryableStreamText(args);
        await Streams.toArray(out.fullStream);
      },
      model: () => MockLanguageModel.from({ doStream: mockStreamChunks }),
      args: { prompt },
    },
    {
      operation: 'embed',
      genAiOperation: 'embeddings',
      call: (args: any) => retryableEmbed(args),
      model: () => MockEmbeddingModel.from([Embedding.vector(3)]),
      args: { value: 'hi' },
    },
    {
      operation: 'embedMany',
      genAiOperation: 'embeddings',
      call: (args: any) => retryableEmbedMany(args),
      model: () => MockEmbeddingModel.from([Embedding.vector(3)]),
      args: { values: ['hi'] },
    },
    {
      operation: 'generateImage',
      genAiOperation: 'generate_content',
      call: (args: any) => retryableGenerateImage(args),
      model: () => MockImageModel.from(mockImageResult),
      args: { prompt: 'a cat' },
    },
  ])(
    'should name the operation span after $operation',
    async ({ operation, genAiOperation, call, model, args }) => {
      // Arrange
      const { exporter, tracer } = createSpanExporter();

      // Act
      await call({
        model: model(),
        ...args,
        retry: { retries: [], telemetry: { isEnabled: true, tracer } },
      });

      // Assert
      const span = findSpan(exporter, `ai_retry.${operation}`);
      expect(span.attributes['ai_retry.operation']).toBe(operation);
      expect(span.attributes['gen_ai.operation.name']).toBe(genAiOperation);
    },
  );
});
