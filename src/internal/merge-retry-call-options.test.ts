import { describe, expect, it } from 'vitest';
import {
  mergeEmbeddingModelCallOptions,
  mergeImageModelCallOptions,
  mergeLanguageModelCallOptions,
} from './merge-retry-call-options.js';
import type {
  EmbeddingModel,
  EmbeddingModelCallOptions,
  ImageModelCallOptions,
  LanguageModel,
  LanguageModelCallOptions,
} from '../types.js';

const samplePrompt: LanguageModelCallOptions['prompt'] = [
  { role: 'user', content: [{ type: 'text', text: 'hello' }] },
];

const overridePrompt: LanguageModelCallOptions['prompt'] = [
  { role: 'user', content: [{ type: 'text', text: 'sanitized' }] },
];

const baseLanguageCallOptions: LanguageModelCallOptions = {
  prompt: samplePrompt,
  temperature: 1,
  topP: 0.9,
  maxOutputTokens: 100,
  providerOptions: { azure: { itemId: 'rs_xyz' } },
};

const baseEmbeddingCallOptions: EmbeddingModelCallOptions = {
  values: ['hello', 'world'],
  providerOptions: { azure: { ref: 'a' } },
};

const baseImageCallOptions: ImageModelCallOptions = {
  prompt: 'a cat',
  n: 1,
  size: '512x512',
  aspectRatio: '1:1',
  seed: 1,
  files: undefined,
  mask: undefined,
  providerOptions: { openai: { quality: 'low' } },
};

describe('mergeLanguageModelCallOptions', () => {
  it('should return base options when no retry or overrides are provided', () => {
    // Arrange / Act
    const result = mergeLanguageModelCallOptions({
      callOptions: baseLanguageCallOptions,
    });

    // Assert
    expect(result.prompt).toBe(samplePrompt);
    expect(result.temperature).toBe(1);
    expect(result.topP).toBe(0.9);
    expect(result.maxOutputTokens).toBe(100);
    expect(result.providerOptions).toEqual({ azure: { itemId: 'rs_xyz' } });
    expect(result.abortSignal).toBeUndefined();
  });

  it('should let Retry.options override base call options', () => {
    // Arrange / Act
    const result = mergeLanguageModelCallOptions({
      callOptions: baseLanguageCallOptions,
      currentRetry: {
        model: {} as LanguageModel,
        options: { temperature: 0.5 },
      },
    });

    // Assert
    expect(result.temperature).toBe(0.5);
    expect(result.topP).toBe(0.9);
  });

  it('should let onRetry overrides beat both base and Retry.options', () => {
    // Arrange / Act
    const result = mergeLanguageModelCallOptions({
      callOptions: baseLanguageCallOptions,
      currentRetry: {
        model: {} as LanguageModel,
        options: { temperature: 0.5 },
      },
      onRetryOverrides: { options: { temperature: 0.1 } },
    });

    // Assert
    expect(result.temperature).toBe(0.1);
  });

  it('should override prompt via onRetry overrides', () => {
    // Arrange / Act
    const result = mergeLanguageModelCallOptions({
      callOptions: baseLanguageCallOptions,
      onRetryOverrides: { options: { prompt: overridePrompt } },
    });

    // Assert
    expect(result.prompt).toBe(overridePrompt);
  });

  describe('providerOptions precedence', () => {
    it('should keep base providerOptions when nothing overrides them', () => {
      // Arrange / Act
      const result = mergeLanguageModelCallOptions({
        callOptions: baseLanguageCallOptions,
      });

      // Assert
      expect(result.providerOptions).toEqual({ azure: { itemId: 'rs_xyz' } });
    });

    it('should use deprecated Retry.providerOptions over base', () => {
      // Arrange / Act
      const result = mergeLanguageModelCallOptions({
        callOptions: baseLanguageCallOptions,
        currentRetry: {
          model: {} as LanguageModel,
          providerOptions: { openai: { reasoningEffort: 'low' } },
        },
      });

      // Assert
      expect(result.providerOptions).toEqual({
        openai: { reasoningEffort: 'low' },
      });
    });

    it('should use Retry.options.providerOptions over deprecated Retry.providerOptions', () => {
      // Arrange / Act
      const result = mergeLanguageModelCallOptions({
        callOptions: baseLanguageCallOptions,
        currentRetry: {
          model: {} as LanguageModel,
          providerOptions: { openai: { reasoningEffort: 'low' } },
          options: { providerOptions: { openai: { reasoningEffort: 'high' } } },
        },
      });

      // Assert
      expect(result.providerOptions).toEqual({
        openai: { reasoningEffort: 'high' },
      });
    });

    it('should use onRetryOverrides.options.providerOptions as the highest precedence', () => {
      // Arrange / Act
      const result = mergeLanguageModelCallOptions({
        callOptions: baseLanguageCallOptions,
        currentRetry: {
          model: {} as LanguageModel,
          providerOptions: { openai: { reasoningEffort: 'low' } },
          options: { providerOptions: { openai: { reasoningEffort: 'high' } } },
        },
        onRetryOverrides: {
          options: {
            providerOptions: { openai: { reasoningEffort: 'minimal' } },
          },
        },
      });

      // Assert
      expect(result.providerOptions).toEqual({
        openai: { reasoningEffort: 'minimal' },
      });
    });
  });

  describe('abortSignal / timeout', () => {
    it('should preserve base abortSignal when no timeout is set', () => {
      // Arrange
      const baseSignal = new AbortController().signal;

      // Act
      const result = mergeLanguageModelCallOptions({
        callOptions: { ...baseLanguageCallOptions, abortSignal: baseSignal },
      });

      // Assert
      expect(result.abortSignal).toBe(baseSignal);
    });

    it('should create a fresh AbortSignal from Retry.timeout', () => {
      // Arrange
      const baseSignal = new AbortController().signal;

      // Act
      const result = mergeLanguageModelCallOptions({
        callOptions: { ...baseLanguageCallOptions, abortSignal: baseSignal },
        currentRetry: { model: {} as LanguageModel, timeout: 30_000 },
      });

      // Assert
      expect(result.abortSignal).toBeDefined();
      expect(result.abortSignal).not.toBe(baseSignal);
      expect(result.abortSignal?.aborted).toBe(false);
    });

    it('should compose base signal with fresh timeout when base is alive', () => {
      // Arrange
      const controller = new AbortController();

      // Act
      const result = mergeLanguageModelCallOptions({
        callOptions: {
          ...baseLanguageCallOptions,
          abortSignal: controller.signal,
        },
        currentRetry: { model: {} as LanguageModel, timeout: 30_000 },
      });

      // Assert
      expect(result.abortSignal).toBeDefined();
      expect(result.abortSignal).not.toBe(controller.signal);
      expect(result.abortSignal?.aborted).toBe(false);

      // Aborting the base should propagate through the composed signal so
      // user cancellation still works mid-retry.
      controller.abort();
      expect(result.abortSignal?.aborted).toBe(true);
    });

    it('should drop already-aborted base TimeoutError when retry timeout is set', async () => {
      // Arrange
      /**
       * Base is an `AbortSignal.timeout` that has already fired (a wall-clock
       * budget the caller passed in that has expired).
       */
      const baseSignal = AbortSignal.timeout(0);
      await new Promise((resolve) => setTimeout(resolve, 5));

      // Act
      const result = mergeLanguageModelCallOptions({
        callOptions: { ...baseLanguageCallOptions, abortSignal: baseSignal },
        currentRetry: { model: {} as LanguageModel, timeout: 30_000 },
      });

      // Assert
      expect(baseSignal.aborted).toBe(true);
      expect(result.abortSignal).toBeDefined();
      expect(result.abortSignal).not.toBe(baseSignal);
      expect(result.abortSignal?.aborted).toBe(false);
    });

    it('should propagate already-aborted base non-TimeoutError when retry timeout is set', () => {
      // Arrange
      const controller = new AbortController();
      controller.abort(new Error('user cancel'));

      // Act
      const result = mergeLanguageModelCallOptions({
        callOptions: {
          ...baseLanguageCallOptions,
          abortSignal: controller.signal,
        },
        currentRetry: { model: {} as LanguageModel, timeout: 30_000 },
      });

      // Assert
      /**
       * User cancellation must still propagate even when the retry has its
       * own timeout. Only deadline-style aborts (TimeoutError) are dropped.
       */
      expect(result.abortSignal?.aborted).toBe(true);
    });

    it('should not propagate base TimeoutError firing mid-retry to composed signal', async () => {
      // Arrange
      /**
       * Base is a wall-clock budget that is about to expire. Caller passed
       * `abortSignal: AbortSignal.timeout(N)` to bound the overall call, not
       * to truncate the per-retry deadline.
       */
      const baseSignal = AbortSignal.timeout(10);

      // Act
      /**
       * Retry slot configures its own per-attempt deadline that is much
       * longer than the base's residual budget.
       */
      const result = mergeLanguageModelCallOptions({
        callOptions: { ...baseLanguageCallOptions, abortSignal: baseSignal },
        currentRetry: { model: {} as LanguageModel, timeout: 30_000 },
      });

      /** Wait past the base deadline, well before the fresh deadline. */
      await new Promise((resolve) => setTimeout(resolve, 50));

      // Assert
      expect(baseSignal.aborted).toBe(true);
      /**
       * A parent `AbortSignal.timeout()` firing is a deadline expiry, not a
       * user cancellation. It must not abort the retry, whose own per-attempt
       * timeout is the deadline the user explicitly configured.
       */
      expect(result.abortSignal?.aborted).toBe(false);
    });

    it('should propagate user cancellation from base even with retry timeout', () => {
      // Arrange
      const controller = new AbortController();

      // Act
      const result = mergeLanguageModelCallOptions({
        callOptions: {
          ...baseLanguageCallOptions,
          abortSignal: controller.signal,
        },
        currentRetry: { model: {} as LanguageModel, timeout: 30_000 },
      });

      /** User cancels with a non-TimeoutError reason. */
      controller.abort(new Error('user cancel'));

      // Assert
      expect(result.abortSignal?.aborted).toBe(true);
    });
  });
});

describe('mergeEmbeddingModelCallOptions', () => {
  it('should return base options when no retry or overrides are provided', () => {
    // Arrange / Act
    const result = mergeEmbeddingModelCallOptions({
      callOptions: baseEmbeddingCallOptions,
    });

    // Assert
    expect(result.values).toEqual(['hello', 'world']);
    expect(result.providerOptions).toEqual({ azure: { ref: 'a' } });
  });

  it('should let onRetry overrides replace values', () => {
    // Arrange / Act
    const result = mergeEmbeddingModelCallOptions({
      callOptions: baseEmbeddingCallOptions,
      onRetryOverrides: { options: { values: ['foo'] } },
    });

    // Assert
    expect(result.values).toEqual(['foo']);
  });

  it('should let Retry.options.providerOptions override base providerOptions', () => {
    // Arrange / Act
    const result = mergeEmbeddingModelCallOptions({
      callOptions: baseEmbeddingCallOptions,
      currentRetry: {
        model: {} as EmbeddingModel,
        options: { providerOptions: { openai: { dimensions: 256 } } },
      },
    });

    // Assert
    expect(result.providerOptions).toEqual({ openai: { dimensions: 256 } });
  });
});

describe('mergeImageModelCallOptions', () => {
  it('should return base options when no retry or overrides are provided', () => {
    // Arrange / Act
    const result = mergeImageModelCallOptions({
      callOptions: baseImageCallOptions,
    });

    // Assert
    expect(result.n).toBe(1);
    expect(result.size).toBe('512x512');
    expect(result.aspectRatio).toBe('1:1');
    expect(result.seed).toBe(1);
    expect(result.providerOptions).toEqual({ openai: { quality: 'low' } });
  });

  it('should let onRetry overrides replace size and seed', () => {
    // Arrange / Act
    const result = mergeImageModelCallOptions({
      callOptions: baseImageCallOptions,
      onRetryOverrides: { options: { size: '1024x1024', seed: 42 } },
    });

    // Assert
    expect(result.size).toBe('1024x1024');
    expect(result.seed).toBe(42);
    expect(result.n).toBe(1);
  });

  it('should keep the required providerOptions when nothing overrides it', () => {
    // Arrange / Act
    const result = mergeImageModelCallOptions({
      callOptions: baseImageCallOptions,
    });

    // Assert
    expect(result.providerOptions).toEqual({ openai: { quality: 'low' } });
  });
});
