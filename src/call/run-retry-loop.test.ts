import { RetryError } from 'ai';
import { describe, expect, it, vi } from 'vitest';
import {
  attemptSpans,
  createSpanExporter,
  findSpan,
  MockLanguageModel,
  mockResultText,
  nonRetryableError,
  retryableError,
} from '../internal/test-utils.js';
import { aborted, finishReason } from './language-model/conditions/index.js';
import { retryableGenerateText } from './language-model/functions/generate-text.js';

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

  it('should report the winning attempt and its result to onSuccess', async () => {
    // Arrange
    const primary = MockLanguageModel.from(retryableError);
    const fallback = MockLanguageModel.from(mockResultText);
    const onSuccess = vi.fn();

    // Act
    await retryableGenerateText({
      model: primary,
      prompt,
      retry: { retries: [fallback], onSuccess },
    });

    // Assert
    const context = onSuccess.mock.calls[0]![0];
    expect(context.current.model).toBe(fallback);
    expect(context.current.result.text).toBe(mockResultText);
    expect(context.current.finishReason).toBe('stop');
    expect(context.attempts.length).toBe(1);
  });

  it('should report a terminal failure to onFailure', async () => {
    // Arrange
    const primary = MockLanguageModel.from(nonRetryableError);
    const onFailure = vi.fn();

    // Act
    const result = retryableGenerateText({
      model: primary,
      prompt,
      retry: { retries: [], onFailure },
    });

    // Assert
    await expect(result).rejects.toThrow();
    const context = onFailure.mock.calls[0]![0];
    expect(context.error).toBe(nonRetryableError);
    expect(context.attempts.length).toBe(1);
  });
});

describe('disabled', () => {
  it('should behave like a direct call', async () => {
    // Arrange
    const primary = MockLanguageModel.from(nonRetryableError);
    const fallback = MockLanguageModel.from(mockResultText);
    const onFailure = vi.fn();

    // Act
    const result = retryableGenerateText({
      model: primary,
      prompt,
      maxRetries: 0,
      retry: { retries: [fallback], disabled: true, onFailure },
    });

    // Assert
    await expect(result).rejects.toThrow();
    expect(fallback.doGenerate.mock.calls.length).toBe(0);
    expect(onFailure.mock.calls.length).toBe(0);
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
});
