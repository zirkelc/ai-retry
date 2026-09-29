import { Errors, Iterables, Streams } from 'ai-test-kit';
import { APICallError, generateText, RetryError, streamText } from 'ai';
import { describe, expect, it, vi } from 'vitest';
import {
  Language,
  MockLanguageModel,
  Options,
  chunksToText,
  contentFilterResult,
  createRetryableModel,
  errorFromChunks,
  errorStreamChunks,
  finishReason,
  mockResult,
  mockResultText,
  nonRetryableError,
  partsToText,
  retryableError,
  successStreamChunks,
} from './test-utils.js';
import type {
  LanguageModel,
  LanguageModelCallOptions,
  LanguageModelStreamPart,
  ModelRetryable,
  RetryableModelOptions,
  ModelRetryContext,
} from '../types.js';
import { isErrorAttempt, isResultAttempt } from './guards.js';

type OnError = Required<RetryableModelOptions<LanguageModel>>['onError'];
type OnRetry = Required<RetryableModelOptions<LanguageModel>>['onRetry'];
type OnSuccess = Required<RetryableModelOptions<LanguageModel>>['onSuccess'];
type OnFailure = Required<RetryableModelOptions<LanguageModel>>['onFailure'];

const prompt = 'Hello!';

const mockStreamChunks: LanguageModelStreamPart[] = [
  Language.streamStart(),
  Language.streamResponseMetadata({
    id: 'id-0',
    modelId: 'mock-model-id',
    timestamp: new Date(0),
  }),
  ...Language.streamText(['Hello', ', ', 'world!'], { id: '1' }),
  Language.streamFinish({
    providerMetadata: {
      testProvider: { testKey: 'testValue' },
    },
  }),
];

/**
 * Stream that finishes with `content-filter` before any text deltas are
 * emitted. The retry layer evaluates result-based retryables against a
 * synthetic result built from the finish part when no content has been
 * streamed yet.
 */
const contentFilterStreamChunks: LanguageModelStreamPart[] = [
  Language.streamStart(),
  Language.streamResponseMetadata({
    id: 'id-0',
    modelId: 'mock-model-id',
    timestamp: new Date(0),
  }),
  Language.streamFinish({
    finishReason: 'content-filter',
    providerMetadata: {
      testProvider: { testKey: 'testValue' },
    },
  }),
];

const refusalText = "I'm sorry, but I cannot assist with that request.";

/**
 * Stream that emits content and *then* finishes with `content-filter`.
 * Once any content has been forwarded downstream, retry would duplicate
 * output, so the finish part flows through unchanged and no retryable is
 * evaluated.
 */
const contentFilterAfterContentStreamChunks: LanguageModelStreamPart[] = [
  Language.streamStart(),
  Language.streamResponseMetadata({
    id: 'id-0',
    modelId: 'mock-model-id',
    timestamp: new Date(0),
  }),
  ...Language.streamText(refusalText, { id: '1' }),
  Language.streamFinish({
    finishReason: 'content-filter',
    providerMetadata: {
      testProvider: { testKey: 'testValue' },
    },
  }),
];

describe('generateText', () => {
  it('should generate text successfully when no errors occur', async () => {
    // Arrange
    const baseModel = MockLanguageModel.from({
      doGenerate: {
        finishReason: { unified: 'stop', raw: undefined },
        usage: {
          inputTokens: { total: 10, noCache: 0, cacheRead: 0, cacheWrite: 0 },
          outputTokens: { total: 20, text: 0, reasoning: 0 },
        },
        content: [{ type: 'text', text: 'Hello, world!' }],
        warnings: [],
      },
    });
    const retryableModel = createRetryableModel({
      model: baseModel,
      retries: [],
    });

    // Act
    const result = await generateText({
      model: retryableModel,
      prompt: 'Hello!',
    });

    // Assert
    expect(baseModel.doGenerate).toHaveBeenCalledTimes(1);
    expect(result.text).toBe('Hello, world!');
  });

  describe('retries', () => {
    describe('error-based retries', () => {
      it('should retry with errors', async () => {
        // Arrange
        const baseModel = MockLanguageModel.from({
          doGenerate: retryableError,
        });
        const fallbackModel = MockLanguageModel.from({
          doGenerate: mockResult,
        });

        const fallbackRetryable: ModelRetryable<LanguageModel> = (context) => {
          if (
            isErrorAttempt(context.current) &&
            APICallError.isInstance(context.current.error)
          ) {
            return { model: fallbackModel, maxAttempts: 1 };
          }
          return undefined;
        };

        // Act
        const result = await generateText({
          model: createRetryableModel({
            model: baseModel,
            retries: [fallbackRetryable],
          }),
          prompt: 'Hello!',
          //
        });

        // Assert
        expect(baseModel.doGenerate).toHaveBeenCalledTimes(1);
        expect(fallbackModel.doGenerate).toHaveBeenCalledTimes(1);
        expect(result.text).toBe(mockResultText);
      });

      it('should not retry without errors', async () => {
        // Arrange
        const baseModel = MockLanguageModel.from({ doGenerate: mockResult });
        const fallbackModel1 = MockLanguageModel.from({
          doGenerate: mockResult,
        });
        const fallbackModel2 = MockLanguageModel.from({
          doGenerate: mockResult,
        });

        // Act
        const result = await generateText({
          model: createRetryableModel({
            model: baseModel,
            retries: [fallbackModel1, fallbackModel2],
          }),
          prompt: 'Hello!',
        });

        // Assert
        expect(baseModel.doGenerate).toHaveBeenCalledTimes(1);
        expect(fallbackModel1.doGenerate).toHaveBeenCalledTimes(0);
        expect(fallbackModel2.doGenerate).toHaveBeenCalledTimes(0);
        expect(result.text).toBe(mockResultText);
      });

      it('should use plain language models for error-based attempts', async () => {
        // Arrange
        const baseModel = MockLanguageModel.from({
          doGenerate: retryableError,
        });
        const fallbackModel1 = MockLanguageModel.from({
          doGenerate: mockResult,
        });
        const fallbackModel2 = MockLanguageModel.from({
          doGenerate: mockResult,
        });

        const fallbackRetryable: ModelRetryable<LanguageModel> = (context) => {
          if (
            isErrorAttempt(context.current) &&
            APICallError.isInstance(context.current.error)
          ) {
            return { model: fallbackModel2, maxAttempts: 1 };
          }
          return undefined;
        };

        // Act
        const result = await generateText({
          model: createRetryableModel({
            model: baseModel,
            retries: [fallbackModel1, fallbackRetryable],
          }),
          prompt: 'Hello!',
        });

        // Assert
        expect(baseModel.doGenerate).toHaveBeenCalledTimes(1);
        expect(fallbackModel1.doGenerate).toHaveBeenCalledTimes(1); // Should be called
        expect(fallbackModel2.doGenerate).toHaveBeenCalledTimes(0);
        expect(result.text).toBe(mockResultText);
      });

      it('should use static retry for error-based attempts', async () => {
        // Arrange
        const baseModel = MockLanguageModel.from({
          doGenerate: retryableError,
        });
        const fallbackModel1 = MockLanguageModel.from({
          doGenerate: mockResult,
        });
        const fallbackModel2 = MockLanguageModel.from({
          doGenerate: mockResult,
        });

        const fallbackRetryable: ModelRetryable<LanguageModel> = (context) => {
          if (
            isErrorAttempt(context.current) &&
            APICallError.isInstance(context.current.error)
          ) {
            return { model: fallbackModel2, maxAttempts: 1 };
          }
          return undefined;
        };

        // Act
        const result = await generateText({
          model: createRetryableModel({
            model: baseModel,
            retries: [{ model: fallbackModel1 }, fallbackRetryable],
          }),
          prompt: 'Hello!',
        });

        // Assert
        expect(baseModel.doGenerate).toHaveBeenCalledTimes(1);
        expect(fallbackModel1.doGenerate).toHaveBeenCalledTimes(1); // Should be called
        expect(fallbackModel2.doGenerate).toHaveBeenCalledTimes(0);
        expect(result.text).toBe(mockResultText);
      });
    });

    describe('result-based retries', () => {
      it('should retry with results', async () => {
        // Arrange
        const baseModel = MockLanguageModel.from({
          doGenerate: contentFilterResult,
        });
        const fallbackModel = MockLanguageModel.from({
          doGenerate: mockResult,
        });
        const fallbackRetryable: ModelRetryable<LanguageModel> = (context) => {
          if (
            isResultAttempt(context.current) &&
            context.current.result.finishReason.unified === 'content-filter'
          ) {
            return { model: fallbackModel, maxAttempts: 1 };
          }
          return undefined;
        };

        // Act
        const result = await generateText({
          model: createRetryableModel({
            model: baseModel,
            retries: [fallbackRetryable],
          }),
          prompt: 'Hello!',
        });

        // Assert
        expect(baseModel.doGenerate).toHaveBeenCalledTimes(1);
        expect(fallbackModel.doGenerate).toHaveBeenCalledTimes(1);
        expect(result.text).toBe(mockResultText);
      });

      it('should ignore plain language models for result-based attempts', async () => {
        // Arrange
        const baseModel = MockLanguageModel.from({
          doGenerate: contentFilterResult,
        });
        const fallbackModel1 = MockLanguageModel.from({
          doGenerate: mockResult,
        });
        const fallbackModel2 = MockLanguageModel.from({
          doGenerate: mockResult,
        });

        const fallbackRetryable: ModelRetryable<LanguageModel> = (context) => {
          if (
            isResultAttempt(context.current) &&
            context.current.result.finishReason.unified === 'content-filter'
          ) {
            return { model: fallbackModel2, maxAttempts: 1 };
          }
          return undefined;
        };

        // Act
        const result = await generateText({
          model: createRetryableModel({
            model: baseModel,
            retries: [
              fallbackModel1, // Language model should be skipped
              fallbackRetryable,
            ],
          }),
          prompt: 'Hello!',
        });

        // Assert
        expect(baseModel.doGenerate).toHaveBeenCalledTimes(1);
        expect(fallbackModel1.doGenerate).toHaveBeenCalledTimes(0); // Should not be called
        expect(fallbackModel2.doGenerate).toHaveBeenCalledTimes(1);
        expect(result.text).toBe(mockResultText);
      });

      it('should ignore static retries for result-based attempts', async () => {
        // Arrange
        const baseModel = MockLanguageModel.from({
          doGenerate: contentFilterResult,
        });
        const fallbackModel1 = MockLanguageModel.from({
          doGenerate: mockResult,
        });
        const fallbackModel2 = MockLanguageModel.from({
          doGenerate: mockResult,
        });

        const fallbackRetryable: ModelRetryable<LanguageModel> = (context) => {
          if (
            isResultAttempt(context.current) &&
            context.current.result.finishReason.unified === 'content-filter'
          ) {
            return { model: fallbackModel2, maxAttempts: 1 };
          }
          return undefined;
        };

        // Act
        const result = await generateText({
          model: createRetryableModel({
            model: baseModel,
            retries: [
              { model: fallbackModel1 }, // Static retry should be skipped
              fallbackRetryable,
            ],
          }),
          prompt: 'Hello!',
        });

        // Assert
        expect(baseModel.doGenerate).toHaveBeenCalledTimes(1);
        expect(fallbackModel1.doGenerate).toHaveBeenCalledTimes(0); // Should not be called
        expect(fallbackModel2.doGenerate).toHaveBeenCalledTimes(1);
        expect(result.text).toBe(mockResultText);
      });
    });
  });

  describe('disabled', () => {
    it('should not retry when disabled is true (boolean)', async () => {
      // Arrange
      const baseModel = MockLanguageModel.from({ doGenerate: retryableError });
      const fallbackModel = MockLanguageModel.from({ doGenerate: mockResult });

      const fallbackRetryable: ModelRetryable<LanguageModel> = () => {
        return { model: fallbackModel, maxAttempts: 1 };
      };

      const retryableModel = createRetryableModel({
        model: baseModel,
        retries: [fallbackRetryable],
        disabled: true,
      });

      // Act & Assert
      await expect(
        generateText({
          model: retryableModel,
          prompt: 'Hello!',
          maxRetries: 0, // Disable AI SDK's own retry mechanism
        }),
      ).rejects.toThrow('Rate limit exceeded');

      expect(baseModel.doGenerate).toHaveBeenCalledTimes(1);
      expect(fallbackModel.doGenerate).toHaveBeenCalledTimes(0);
    });

    it('should retry when disabled is false (boolean)', async () => {
      // Arrange
      const baseModel = MockLanguageModel.from({ doGenerate: retryableError });
      const fallbackModel = MockLanguageModel.from({ doGenerate: mockResult });

      const fallbackRetryable: ModelRetryable<LanguageModel> = () => {
        return { model: fallbackModel, maxAttempts: 1 };
      };

      const retryableModel = createRetryableModel({
        model: baseModel,
        retries: [fallbackRetryable],
        disabled: false,
      });

      // Act
      const result = await generateText({
        model: retryableModel,
        prompt: 'Hello!',
      });

      // Assert
      expect(baseModel.doGenerate).toHaveBeenCalledTimes(1);
      expect(fallbackModel.doGenerate).toHaveBeenCalledTimes(1);
      expect(result.text).toBe('Hello, world!');
    });

    it('should not retry when disabled is a function returning true', async () => {
      // Arrange
      const baseModel = MockLanguageModel.from({ doGenerate: retryableError });
      const fallbackModel = MockLanguageModel.from({ doGenerate: mockResult });

      const fallbackRetryable: ModelRetryable<LanguageModel> = () => {
        return { model: fallbackModel, maxAttempts: 1 };
      };

      const disabledFn = vi.fn(() => true);

      const retryableModel = createRetryableModel({
        model: baseModel,
        retries: [fallbackRetryable],
        disabled: disabledFn,
      });

      // Act & Assert
      await expect(
        generateText({
          model: retryableModel,
          prompt: 'Hello!',
          maxRetries: 0, // Disable AI SDK's own retry mechanism
        }),
      ).rejects.toThrow('Rate limit exceeded');

      expect(disabledFn).toHaveBeenCalled();
      expect(baseModel.doGenerate).toHaveBeenCalledTimes(1);
      expect(fallbackModel.doGenerate).toHaveBeenCalledTimes(0);
    });

    it('should retry when disabled is a function returning false', async () => {
      // Arrange
      const baseModel = MockLanguageModel.from({ doGenerate: retryableError });
      const fallbackModel = MockLanguageModel.from({ doGenerate: mockResult });

      const fallbackRetryable: ModelRetryable<LanguageModel> = () => {
        return { model: fallbackModel, maxAttempts: 1 };
      };

      const disabledFn = vi.fn(() => false);

      const retryableModel = createRetryableModel({
        model: baseModel,
        retries: [fallbackRetryable],
        disabled: disabledFn,
      });

      // Act
      const result = await generateText({
        model: retryableModel,
        prompt: 'Hello!',
      });

      // Assert
      expect(disabledFn).toHaveBeenCalled();
      expect(baseModel.doGenerate).toHaveBeenCalledTimes(1);
      expect(fallbackModel.doGenerate).toHaveBeenCalledTimes(1);
      expect(result.text).toBe('Hello, world!');
    });

    it('should work normally when disabled is undefined (default)', async () => {
      // Arrange
      const baseModel = MockLanguageModel.from({ doGenerate: retryableError });
      const fallbackModel = MockLanguageModel.from({ doGenerate: mockResult });

      const fallbackRetryable: ModelRetryable<LanguageModel> = () => {
        return { model: fallbackModel, maxAttempts: 1 };
      };

      const retryableModel = createRetryableModel({
        model: baseModel,
        retries: [fallbackRetryable],
        // disabled is undefined by default
      });

      // Act
      const result = await generateText({
        model: retryableModel,
        prompt: 'Hello!',
      });

      // Assert
      expect(baseModel.doGenerate).toHaveBeenCalledTimes(1);
      expect(fallbackModel.doGenerate).toHaveBeenCalledTimes(1);
      expect(result.text).toBe('Hello, world!');
    });
  });

  describe('onError', () => {
    it('should call onError handler when an error occurs', async () => {
      // Arrange
      const baseModel = MockLanguageModel.from({ doGenerate: retryableError });
      const fallbackModel = MockLanguageModel.from({ doGenerate: mockResult });
      const onErrorSpy = vi.fn<OnError>();

      // Act
      await generateText({
        model: createRetryableModel({
          model: baseModel,
          retries: [fallbackModel],
          onError: onErrorSpy,
        }),
        prompt: 'Hello!',
      });

      // Assert
      expect(onErrorSpy).toHaveBeenCalledTimes(1);

      const firstErrorCall = onErrorSpy.mock.calls[0]![0];
      expect(firstErrorCall.current.error).toBe(retryableError);
      expect(firstErrorCall.current.model).toBe(baseModel);
      expect(firstErrorCall.attempts.length).toBe(1);
      // expect(firstErrorCall.totalAttempts).toBe(1);
    });

    it('should call onError handler for each error in multiple attempts', async () => {
      // Arrange
      const baseModel = MockLanguageModel.from({ doGenerate: retryableError });
      const fallbackModel = MockLanguageModel.from({
        doGenerate: nonRetryableError,
      });
      const finalModel = MockLanguageModel.from({ doGenerate: mockResult });
      const onErrorSpy = vi.fn<OnError>();

      // Act
      await generateText({
        model: createRetryableModel({
          model: baseModel,
          retries: [fallbackModel, finalModel],
          onError: onErrorSpy,
        }),
        prompt: 'Hello!',
      });

      // Assert
      expect(onErrorSpy).toHaveBeenCalledTimes(2);

      // Check that onError was called for each error
      const firstErrorCall = onErrorSpy.mock.calls[0]![0];
      const secondErrorCall = onErrorSpy.mock.calls[1]![0];

      expect(firstErrorCall.current.error).toBe(retryableError);
      expect(firstErrorCall.current.model).toBe(baseModel);
      expect(firstErrorCall.attempts.length).toBe(1);
      // expect(firstErrorCall.totalAttempts).toBe(1);

      expect(secondErrorCall.current.error).toBe(nonRetryableError);
      expect(secondErrorCall.current.model).toBe(fallbackModel);
      expect(secondErrorCall.attempts.length).toBe(2);
      // expect(secondErrorCall.totalAttempts).toBe(2);
    });

    it('should NOT call onError handler for result-based retries', async () => {
      // Arrange
      const baseModel = MockLanguageModel.from({
        doGenerate: contentFilterResult,
      });
      const fallbackModel = MockLanguageModel.from({ doGenerate: mockResult });
      const onErrorSpy = vi.fn<OnError>();
      const onRetrySpy = vi.fn<OnRetry>();

      // Act
      await generateText({
        model: createRetryableModel({
          model: baseModel,
          retries: [
            (context: ModelRetryContext<LanguageModel>) => {
              if (
                isResultAttempt(context.current) &&
                context.current.result.finishReason.unified === 'content-filter'
              ) {
                return { model: fallbackModel, maxAttempts: 1 };
              }
              return undefined;
            },
          ],
          onError: onErrorSpy,
          onRetry: onRetrySpy,
        }),
        prompt: 'Hello!',
      });

      // Assert
      expect(onErrorSpy).not.toHaveBeenCalled();
      expect(onRetrySpy).toHaveBeenCalledTimes(1);
    });

    it('should call onError handler before onRetry handler', async () => {
      // Arrange
      const baseModel = MockLanguageModel.from({ doGenerate: retryableError });
      const fallbackModel = MockLanguageModel.from({ doGenerate: mockResult });
      const onErrorSpy = vi.fn<OnError>();
      const onRetrySpy = vi.fn<OnRetry>();

      // Act
      await generateText({
        model: createRetryableModel({
          model: baseModel,
          retries: [fallbackModel],
          onError: onErrorSpy,
          onRetry: onRetrySpy,
        }),
        prompt: 'Hello!',
      });

      // Assert
      expect(onErrorSpy).toHaveBeenCalledTimes(1);
      expect(onRetrySpy).toHaveBeenCalledTimes(1);

      // Verify onError is called before onRetry by checking call order
      const errorCallTime = onErrorSpy.mock.invocationCallOrder[0] ?? 0;
      const retryCallTime = onRetrySpy.mock.invocationCallOrder[0] ?? 0;
      expect(errorCallTime).toBeLessThan(retryCallTime);

      // Verify the context passed to each handler
      const firstErrorCall = onErrorSpy.mock.calls[0]![0];
      const firstRetryCall = onRetrySpy.mock.calls[0]![0];
      expect(firstErrorCall.current.model).toBe(baseModel);
      expect(firstErrorCall.attempts.length).toBe(1);
      expect(firstRetryCall.current.model).toBe(fallbackModel);
      expect(firstRetryCall.attempts.length).toBe(1);
    });
  });

  describe('onRetry', () => {
    it('should call onRetry handler for error-based retries', async () => {
      // Arrange
      const baseModel = MockLanguageModel.from({ doGenerate: retryableError });
      const fallbackModel = MockLanguageModel.from({ doGenerate: mockResult });
      const onRetrySpy = vi.fn<OnRetry>();

      // Act
      await generateText({
        model: createRetryableModel({
          model: baseModel,
          retries: [fallbackModel],
          onRetry: onRetrySpy,
        }),
        prompt: 'Hello!',
      });

      // Assert
      expect(onRetrySpy).toHaveBeenCalledTimes(1);

      const firstRetryCall = onRetrySpy.mock.calls[0]![0];
      expect(isErrorAttempt(firstRetryCall.current)).toBe(true);
      if (isErrorAttempt(firstRetryCall.current)) {
        expect(firstRetryCall.current.error).toBe(retryableError);
      }
      expect(firstRetryCall.current.model).toBe(fallbackModel);
      expect(firstRetryCall.attempts.length).toBe(1);
      // expect(firstRetryCall.totalAttempts).toBe(1);
    });

    it('should call onRetry handler for result-based retries', async () => {
      // Arrange
      const baseModel = MockLanguageModel.from({
        doGenerate: contentFilterResult,
      });
      const fallbackModel = MockLanguageModel.from({ doGenerate: mockResult });
      const onRetrySpy = vi.fn<OnRetry>();

      // Act
      await generateText({
        model: createRetryableModel({
          model: baseModel,
          retries: [
            (context: ModelRetryContext<LanguageModel>) => {
              if (
                isResultAttempt(context.current) &&
                context.current.result.finishReason.unified === 'content-filter'
              ) {
                return { model: fallbackModel, maxAttempts: 1 };
              }
              return undefined;
            },
          ],
          onRetry: onRetrySpy,
        }),
        prompt: 'Hello!',
      });

      // Assert
      expect(onRetrySpy).toHaveBeenCalledTimes(1);

      const retryCall = onRetrySpy.mock.calls[0]![0];
      expect(isResultAttempt(retryCall.current)).toBe(true);
      if (isResultAttempt(retryCall.current)) {
        expect(retryCall.current.result.finishReason.unified).toBe(
          'content-filter',
        );
      }
      expect(retryCall.current.model).toBe(fallbackModel);
      expect(retryCall.attempts.length).toBe(1);
      // expect(retryCall.totalAttempts).toBe(1);
    });

    it('should call onRetry handler for each retry attempt', async () => {
      // Arrange
      const baseModel = MockLanguageModel.from({ doGenerate: retryableError });
      const fallbackModel1 = MockLanguageModel.from({
        doGenerate: nonRetryableError,
      });
      const fallbackModel2 = MockLanguageModel.from({ doGenerate: mockResult });
      const onRetrySpy = vi.fn<OnRetry>();

      // Act
      await generateText({
        model: createRetryableModel({
          model: baseModel,
          retries: [fallbackModel1, fallbackModel2],
          onRetry: onRetrySpy,
        }),
        prompt: 'Hello!',
      });

      // Assert
      expect(onRetrySpy).toHaveBeenCalledTimes(2);

      // Check that onRetry was called for each retry
      const firstRetryCall = onRetrySpy.mock.calls[0]![0];
      const secondRetryCall = onRetrySpy.mock.calls[1]![0];

      expect(isErrorAttempt(firstRetryCall.current)).toBe(true);
      if (isErrorAttempt(firstRetryCall.current)) {
        expect(firstRetryCall.current.error).toBe(retryableError);
      }
      expect(firstRetryCall.current.model).toBe(fallbackModel1);
      expect(firstRetryCall.attempts.length).toBe(1);
      // expect(firstRetryCall.totalAttempts).toBe(1);

      expect(isErrorAttempt(secondRetryCall.current)).toBe(true);
      if (isErrorAttempt(secondRetryCall.current)) {
        expect(secondRetryCall.current.error).toBe(nonRetryableError);
      }
      expect(secondRetryCall.current.model).toBe(fallbackModel2);
      expect(secondRetryCall.attempts.length).toBe(2);
      // expect(secondRetryCall.totalAttempts).toBe(2);
    });

    it('should NOT call onRetry on first attempt', async () => {
      // Arrange
      const baseModel = MockLanguageModel.from({ doGenerate: mockResult });
      const onRetrySpy = vi.fn<OnRetry>();

      // Act
      await generateText({
        model: createRetryableModel({
          model: baseModel,
          retries: [],
          onRetry: onRetrySpy,
        }),
        prompt: 'Hello!',
      });

      // Assert
      expect(onRetrySpy).not.toHaveBeenCalled();
    });

    describe('overrides', () => {
      it('should override prompt for the upcoming retry attempt', async () => {
        // Arrange
        const baseModel = MockLanguageModel.from({
          doGenerate: retryableError,
        });
        const fallbackModel = MockLanguageModel.from({
          doGenerate: mockResult,
        });
        const sanitizedPrompt = [
          {
            role: 'user' as const,
            content: [{ type: 'text' as const, text: 'sanitized' }],
          },
        ];

        // Act
        await generateText({
          model: createRetryableModel({
            model: baseModel,
            retries: [fallbackModel],
            onRetry: () => ({ options: { prompt: sanitizedPrompt } }),
          }),
          prompt: 'original',
        });

        // Assert
        expect(fallbackModel.doGenerate).toHaveBeenCalledWith(
          expect.objectContaining({ prompt: sanitizedPrompt }),
        );
      });

      it('should override providerOptions for the upcoming retry attempt', async () => {
        // Arrange — simulates stripping provider-scoped metadata when
        // crossing a provider boundary on retry.
        const baseModel = MockLanguageModel.from({
          doGenerate: retryableError,
        });
        const fallbackModel = MockLanguageModel.from({
          doGenerate: mockResult,
        });
        const sanitizedProviderOptions = { openai: { reasoningEffort: 'low' } };

        // Act
        await generateText({
          model: createRetryableModel({
            model: baseModel,
            retries: [fallbackModel],
            onRetry: () => ({
              options: { providerOptions: sanitizedProviderOptions },
            }),
          }),
          prompt: 'Hello!',
          providerOptions: { azure: { itemId: 'rs_xyz' } },
        });

        // Assert
        expect(fallbackModel.doGenerate).toHaveBeenCalledWith(
          expect.objectContaining({
            providerOptions: sanitizedProviderOptions,
          }),
        );
      });

      it('should let onRetry overrides beat Retry.options', async () => {
        // Arrange
        const baseModel = MockLanguageModel.from({
          doGenerate: retryableError,
        });
        const fallbackModel = MockLanguageModel.from({
          doGenerate: mockResult,
        });

        // Act
        await generateText({
          model: createRetryableModel({
            model: baseModel,
            retries: [{ model: fallbackModel, options: { temperature: 0.5 } }],
            onRetry: () => ({ options: { temperature: 0.1 } }),
          }),
          prompt: 'Hello!',
          temperature: 1.0,
        });

        // Assert
        expect(fallbackModel.doGenerate).toHaveBeenCalledWith(
          expect.objectContaining({ temperature: 0.1 }),
        );
      });

      it('should fall back to Retry.options when onRetry returns undefined', async () => {
        // Arrange
        const baseModel = MockLanguageModel.from({
          doGenerate: retryableError,
        });
        const fallbackModel = MockLanguageModel.from({
          doGenerate: mockResult,
        });

        // Act
        await generateText({
          model: createRetryableModel({
            model: baseModel,
            retries: [{ model: fallbackModel, options: { temperature: 0.5 } }],
            onRetry: () => undefined,
          }),
          prompt: 'Hello!',
          temperature: 1.0,
        });

        // Assert
        expect(fallbackModel.doGenerate).toHaveBeenCalledWith(
          expect.objectContaining({ temperature: 0.5 }),
        );
      });

      it('should support async onRetry returning overrides', async () => {
        // Arrange
        const baseModel = MockLanguageModel.from({
          doGenerate: retryableError,
        });
        const fallbackModel = MockLanguageModel.from({
          doGenerate: mockResult,
        });

        // Act
        await generateText({
          model: createRetryableModel({
            model: baseModel,
            retries: [fallbackModel],
            onRetry: async () => {
              await Promise.resolve();
              return { options: { temperature: 0.42 } };
            },
          }),
          prompt: 'Hello!',
        });

        // Assert
        expect(fallbackModel.doGenerate).toHaveBeenCalledWith(
          expect.objectContaining({ temperature: 0.42 }),
        );
      });
    });
  });

  describe('onSuccess', () => {
    it('should call onSuccess with base model when no retry occurs', async () => {
      // Arrange
      const baseModel = MockLanguageModel.from({ doGenerate: mockResult });
      const onSuccessSpy = vi.fn<OnSuccess>();

      // Act
      await generateText({
        model: createRetryableModel({
          model: baseModel,
          retries: [],
          onSuccess: onSuccessSpy,
        }),
        prompt: 'Hello!',
      });

      // Assert
      expect(onSuccessSpy).toHaveBeenCalledTimes(1);

      const successCall = onSuccessSpy.mock.calls[0]![0];
      expect(successCall.current.type).toBe('success');
      expect(successCall.current.model).toBe(baseModel);
      expect(successCall.current.result).toBeDefined();
      expect(successCall.attempts.length).toBe(0);
    });

    it('should call onSuccess with fallback model after retry', async () => {
      // Arrange
      const baseModel = MockLanguageModel.from({ doGenerate: retryableError });
      const fallbackModel = MockLanguageModel.from({ doGenerate: mockResult });
      const onSuccessSpy = vi.fn<OnSuccess>();

      // Act
      await generateText({
        model: createRetryableModel({
          model: baseModel,
          retries: [fallbackModel],
          onSuccess: onSuccessSpy,
        }),
        prompt: 'Hello!',
      });

      // Assert
      expect(onSuccessSpy).toHaveBeenCalledTimes(1);

      const successCall = onSuccessSpy.mock.calls[0]![0];
      expect(successCall.current.type).toBe('success');
      expect(successCall.current.model).toBe(fallbackModel);
      expect(successCall.current.result).toBeDefined();
      expect(successCall.attempts.length).toBe(1);
    });

    it('should call onSuccess with the retried result attempt after a result-based retry', async () => {
      // Arrange
      const baseModel = MockLanguageModel.from({
        doGenerate: contentFilterResult,
      });
      const fallbackModel = MockLanguageModel.from({ doGenerate: mockResult });
      const fallbackRetryable: ModelRetryable<LanguageModel> = (context) => {
        if (
          isResultAttempt(context.current) &&
          context.current.result.finishReason.unified === 'content-filter'
        ) {
          return { model: fallbackModel, maxAttempts: 1 };
        }
        return undefined;
      };
      const onSuccessSpy = vi.fn<OnSuccess>();

      // Act
      await generateText({
        model: createRetryableModel({
          model: baseModel,
          retries: [fallbackRetryable],
          onSuccess: onSuccessSpy,
        }),
        prompt: 'Hello!',
      });

      // Assert
      const successCall = onSuccessSpy.mock.calls[0]![0];
      expect(successCall.current.model).toBe(fallbackModel);
      expect(successCall.attempts.length).toBe(1);
      expect(successCall.attempts[0]!.type).toBe('result');
      expect(successCall.attempts[0]!.model).toBe(baseModel);
    });

    it('should NOT call onSuccess when all retries fail', async () => {
      // Arrange
      const baseModel = MockLanguageModel.from({ doGenerate: retryableError });
      const fallbackModel = MockLanguageModel.from({
        doGenerate: nonRetryableError,
      });
      const onSuccessSpy = vi.fn<OnSuccess>();

      // Act & Assert
      const result = generateText({
        model: createRetryableModel({
          model: baseModel,
          retries: [fallbackModel],
          onSuccess: onSuccessSpy,
        }),
        prompt: 'Hello!',
      });
      await expect(result).rejects.toThrow();

      expect(onSuccessSpy).not.toHaveBeenCalled();
    });
  });

  describe('onFailure', () => {
    it('should call onFailure with raw error when no retry is available', async () => {
      // Arrange
      const baseModel = MockLanguageModel.from({
        doGenerate: nonRetryableError,
      });
      const onFailureSpy = vi.fn<OnFailure>();

      // Act
      const result = generateText({
        model: createRetryableModel({
          model: baseModel,
          retries: [],
          onFailure: onFailureSpy,
        }),
        prompt,
      });
      await expect(result).rejects.toThrow();

      // Assert
      expect(onFailureSpy).toHaveBeenCalledTimes(1);

      const failureCall = onFailureSpy.mock.calls[0]![0];
      expect(failureCall.current.type).toBe('error');
      expect(failureCall.current.model).toBe(baseModel);
      expect(failureCall.attempts.length).toBe(1);
      expect(failureCall.error).toBe(nonRetryableError);
    });

    it('should call onFailure with RetryError when retries are exhausted', async () => {
      // Arrange
      const baseModel = MockLanguageModel.from({ doGenerate: retryableError });
      const fallbackModel = MockLanguageModel.from({
        doGenerate: nonRetryableError,
      });
      const onFailureSpy = vi.fn<OnFailure>();

      // Act
      const result = generateText({
        model: createRetryableModel({
          model: baseModel,
          retries: [fallbackModel],
          onFailure: onFailureSpy,
        }),
        prompt,
      });
      await expect(result).rejects.toThrow();

      // Assert
      expect(onFailureSpy).toHaveBeenCalledTimes(1);

      const failureCall = onFailureSpy.mock.calls[0]![0];
      expect(failureCall.current.type).toBe('error');
      expect(failureCall.current.model).toBe(fallbackModel);
      expect(failureCall.attempts.length).toBe(2);
      expect(failureCall.error).toBeInstanceOf(RetryError);
    });

    it('should NOT call onFailure on success', async () => {
      // Arrange
      const baseModel = MockLanguageModel.from({ doGenerate: mockResult });
      const onFailureSpy = vi.fn<OnFailure>();
      const onSuccessSpy = vi.fn<OnSuccess>();

      // Act
      await generateText({
        model: createRetryableModel({
          model: baseModel,
          retries: [],
          onFailure: onFailureSpy,
          onSuccess: onSuccessSpy,
        }),
        prompt,
      });

      // Assert
      expect(onSuccessSpy).toHaveBeenCalledTimes(1);
      expect(onFailureSpy).not.toHaveBeenCalled();
    });

    it('should NOT call onFailure when an onSuccess handler throws', async () => {
      // Arrange — a throwing hook is not an attempt failure. The request
      // succeeded, so reporting it as a failure would fire both callbacks
      // for one request, with a stale attempt as `current`.
      /**
       * Explicit model ids so the shared auto-increment counter is untouched
       * and the inline snapshots further down the file stay stable.
       */
      const baseModel = MockLanguageModel.from(
        { doGenerate: retryableError },
        { modelId: 'throwing-on-success-base' },
      );
      const fallbackModel = MockLanguageModel.from(
        { doGenerate: mockResult },
        { modelId: 'throwing-on-success-fallback' },
      );
      const handlerError = new Error('onSuccess blew up');
      const onFailureSpy = vi.fn<OnFailure>();
      const onSuccessSpy = vi.fn<OnSuccess>(() => {
        throw handlerError;
      });

      // Act
      const result = generateText({
        model: createRetryableModel({
          model: baseModel,
          retries: [fallbackModel],
          onFailure: onFailureSpy,
          onSuccess: onSuccessSpy,
        }),
        prompt,
      });
      await expect(result).rejects.toThrow();

      // Assert — the handler error surfaces unwrapped.
      await result.catch((e) => expect(e).toBe(handlerError));
      expect(onSuccessSpy).toHaveBeenCalledTimes(1);
      expect(onFailureSpy).not.toHaveBeenCalled();
    });

    it('should NOT call onSuccess on failure', async () => {
      // Arrange
      const baseModel = MockLanguageModel.from({ doGenerate: retryableError });
      const fallbackModel = MockLanguageModel.from({
        doGenerate: nonRetryableError,
      });
      const onFailureSpy = vi.fn<OnFailure>();
      const onSuccessSpy = vi.fn<OnSuccess>();

      // Act
      const result = generateText({
        model: createRetryableModel({
          model: baseModel,
          retries: [fallbackModel],
          onFailure: onFailureSpy,
          onSuccess: onSuccessSpy,
        }),
        prompt,
      });
      await expect(result).rejects.toThrow();

      // Assert
      expect(onFailureSpy).toHaveBeenCalledTimes(1);
      expect(onSuccessSpy).not.toHaveBeenCalled();
    });
  });

  describe('attempt options', () => {
    it('should include call options in error attempts', async () => {
      // Arrange
      const baseModel = MockLanguageModel.from({ doGenerate: retryableError });
      const fallbackModel = MockLanguageModel.from({ doGenerate: mockResult });
      const onErrorSpy = vi.fn<OnError>();

      // Act
      await generateText({
        model: createRetryableModel({
          model: baseModel,
          retries: [fallbackModel],
          onError: onErrorSpy,
        }),
        prompt: 'Hello!',
        temperature: 0.7,
        maxOutputTokens: 1000,
      });

      // Assert
      expect(onErrorSpy).toHaveBeenCalledTimes(1);

      const errorContext = onErrorSpy.mock.calls[0]![0];
      expect(errorContext.current.options).toBeDefined();
      expect(errorContext.current.options.temperature).toBe(0.7);
      expect(errorContext.current.options.maxOutputTokens).toBe(1000);
    });

    it('should include call options in result attempts', async () => {
      // Arrange
      const baseModel = MockLanguageModel.from({
        doGenerate: contentFilterResult,
      });
      const fallbackModel = MockLanguageModel.from({ doGenerate: mockResult });
      const onRetrySpy = vi.fn<OnRetry>();

      // Act
      await generateText({
        model: createRetryableModel({
          model: baseModel,
          retries: [
            (context) => {
              if (
                isResultAttempt(context.current) &&
                context.current.result.finishReason.unified === 'content-filter'
              ) {
                return { model: fallbackModel, maxAttempts: 1 };
              }
              return undefined;
            },
          ],
          onRetry: onRetrySpy,
        }),
        prompt: 'Hello!',
        temperature: 0.8,
        seed: 42,
      });

      // Assert
      expect(onRetrySpy).toHaveBeenCalledTimes(1);

      const retryContext = onRetrySpy.mock.calls[0]![0];
      expect(isResultAttempt(retryContext.current)).toBe(true);
      if (isResultAttempt(retryContext.current)) {
        expect(retryContext.current.options).toBeDefined();
        expect(retryContext.current.options.temperature).toBe(0.8);
        expect(retryContext.current.options.seed).toBe(42);
      }
    });

    it('should reflect overridden options in retry attempts', async () => {
      // Arrange
      const baseModel = MockLanguageModel.from({ doGenerate: retryableError });
      const fallbackModel = MockLanguageModel.from({
        doGenerate: nonRetryableError,
      });
      const finalModel = MockLanguageModel.from({ doGenerate: mockResult });
      const onErrorSpy = vi.fn<OnError>();

      // Act
      await generateText({
        model: createRetryableModel({
          model: baseModel,
          retries: [
            { model: fallbackModel, options: { temperature: 0.5 } },
            finalModel,
          ],
          onError: onErrorSpy,
        }),
        prompt: 'Hello!',
        temperature: 1.0,
      });

      // Assert
      expect(onErrorSpy).toHaveBeenCalledTimes(2);

      // First attempt should have original temperature
      const firstErrorContext = onErrorSpy.mock.calls[0]![0];
      expect(firstErrorContext.current.options.temperature).toBe(1.0);

      // Second attempt should have overridden temperature
      const secondErrorContext = onErrorSpy.mock.calls[1]![0];
      expect(secondErrorContext.current.options.temperature).toBe(0.5);
    });

    it('should include prompt in options', async () => {
      // Arrange
      const baseModel = MockLanguageModel.from({ doGenerate: retryableError });
      const fallbackModel = MockLanguageModel.from({ doGenerate: mockResult });
      const onErrorSpy = vi.fn<OnError>();

      // Act
      await generateText({
        model: createRetryableModel({
          model: baseModel,
          retries: [fallbackModel],
          onError: onErrorSpy,
        }),
        prompt: 'Test prompt',
      });

      // Assert
      expect(onErrorSpy).toHaveBeenCalledTimes(1);

      const errorContext = onErrorSpy.mock.calls[0]![0];
      expect(errorContext.current.options.prompt).toBeDefined();
      expect(errorContext.current.options.prompt).toEqual(
        expect.arrayContaining([
          expect.objectContaining({
            role: 'user',
            content: expect.arrayContaining([
              expect.objectContaining({ type: 'text', text: 'Test prompt' }),
            ]),
          }),
        ]),
      );
    });
  });

  describe('RetryableOptions', () => {
    describe('maxAttempts', () => {
      it('should try each model only once by default', async () => {
        // Arrange
        const baseModel = MockLanguageModel.from({
          doGenerate: retryableError,
        });
        const fallbackModel1 = MockLanguageModel.from({
          doGenerate: retryableError,
        });
        const fallbackModel2 = MockLanguageModel.from({
          doGenerate: retryableError,
        });
        const finalModel = MockLanguageModel.from({ doGenerate: mockResult });

        // Act
        await generateText({
          model: createRetryableModel({
            model: baseModel,
            retries: [
              fallbackModel1,
              { model: fallbackModel2 },
              async () => ({ model: finalModel }),
            ],
          }),
          prompt: 'Hello!',
        });

        // Assert
        expect(baseModel.doGenerate).toHaveBeenCalledTimes(1);
        expect(fallbackModel1.doGenerate).toHaveBeenCalledTimes(1);
        expect(fallbackModel2.doGenerate).toHaveBeenCalledTimes(1);
        expect(finalModel.doGenerate).toHaveBeenCalledTimes(1);
      });

      it('should try models multiple times if maxAttempts is set', async () => {
        // Arrange
        const baseModel = MockLanguageModel.from({
          doGenerate: retryableError,
        });
        const fallbackModel1 = MockLanguageModel.from({
          doGenerate: retryableError,
        });
        const fallbackModel2 = MockLanguageModel.from({
          doGenerate: retryableError,
        });
        const finalModel = MockLanguageModel.from({ doGenerate: mockResult });

        // Act
        await generateText({
          model: createRetryableModel({
            model: baseModel,
            retries: [
              // ModelRetryable<LanguageModel>  with different maxAttempts
              { model: fallbackModel1, maxAttempts: 2 },
              async () => ({ model: fallbackModel2, maxAttempts: 3 }),
              finalModel,
            ],
          }),
          prompt: 'Hello!',
        });

        // Assert
        expect(baseModel.doGenerate).toHaveBeenCalledTimes(1);
        expect(fallbackModel1.doGenerate).toHaveBeenCalledTimes(2);
        expect(fallbackModel2.doGenerate).toHaveBeenCalledTimes(3);
        expect(finalModel.doGenerate).toHaveBeenCalledTimes(1);
      });
    });

    describe('maxRetries', () => {
      it('should ignore maxRetries setting when retryable matches', async () => {
        // Arrange
        const baseModel = MockLanguageModel.from({
          doGenerate: retryableError,
        });
        const fallbackModel = MockLanguageModel.from({
          doGenerate: nonRetryableError,
        });

        // Act & Assert
        try {
          await generateText({
            model: createRetryableModel({
              model: baseModel,
              retries: [fallbackModel],
            }),
            prompt: 'Hello!',
            maxRetries: 1, // Should be ignored since RetryError is thrown
          });
          expect.unreachable('Should throw RetryError');
        } catch (error) {
          expect(error).toBeInstanceOf(RetryError);
          expect(baseModel.doGenerate).toHaveBeenCalledTimes(1);
          expect(fallbackModel.doGenerate).toHaveBeenCalledTimes(1);
        }
      });

      it('should respect maxRetries setting when no retryable matches', async () => {
        // Arrange
        const baseModel = MockLanguageModel.from({
          doGenerate: retryableError,
        });

        // Act & Assert
        try {
          await generateText({
            model: createRetryableModel({
              model: baseModel,
              retries: [],
            }),
            prompt: 'Hello!',
            maxRetries: 1, // Should be ignored since RetryError is thrown
          });
          expect.unreachable('Should throw RetryError');
        } catch (error) {
          expect(error).toBeInstanceOf(RetryError);
          expect(baseModel.doGenerate).toHaveBeenCalledTimes(2); // 1 initial + 1 retry
        }
      });
    });

    describe('delay', () => {
      it('should apply delay before retrying', async () => {
        // Arrange
        vi.useFakeTimers();
        const baseModel = MockLanguageModel.from({
          doGenerate: retryableError,
        });
        const fallbackModel = MockLanguageModel.from({
          doGenerate: mockResult,
        });
        const delayMs = 100;

        // Act
        const promise = generateText({
          model: createRetryableModel({
            model: baseModel,
            retries: [{ model: fallbackModel, delay: delayMs }],
          }),
          prompt: 'Hello!',
        });

        await vi.runAllTimersAsync();
        await promise;

        // Assert
        expect(baseModel.doGenerate).toHaveBeenCalledTimes(1);
        expect(fallbackModel.doGenerate).toHaveBeenCalledTimes(1);

        vi.useRealTimers();
      });

      it('should apply different delays for multiple retries', async () => {
        // Arrange
        vi.useFakeTimers();
        const baseModel = MockLanguageModel.from({
          doGenerate: retryableError,
        });
        const fallbackModel1 = MockLanguageModel.from({
          doGenerate: retryableError,
        });
        const fallbackModel2 = MockLanguageModel.from({
          doGenerate: mockResult,
        });
        const delay1 = 50;
        const delay2 = 50;

        // Act
        const promise = generateText({
          model: createRetryableModel({
            model: baseModel,
            retries: [
              { model: fallbackModel1, delay: delay1 },
              () => ({ model: fallbackModel2, delay: delay2 }),
            ],
          }),
          prompt: 'Hello!',
        });

        await vi.runAllTimersAsync();
        await promise;

        // Assert
        expect(baseModel.doGenerate).toHaveBeenCalledTimes(1);
        expect(fallbackModel1.doGenerate).toHaveBeenCalledTimes(1);
        expect(fallbackModel2.doGenerate).toHaveBeenCalledTimes(1);

        vi.useRealTimers();
      });

      it('should not delay when delay is not specified', async () => {
        // Arrange
        const baseModel = MockLanguageModel.from({
          doGenerate: retryableError,
        });
        const fallbackModel = MockLanguageModel.from({
          doGenerate: mockResult,
        });

        // Act
        await generateText({
          model: createRetryableModel({
            model: baseModel,
            retries: [{ model: fallbackModel }],
          }),
          prompt: 'Hello!',
        });

        // Assert
        expect(baseModel.doGenerate).toHaveBeenCalledTimes(1);
        expect(fallbackModel.doGenerate).toHaveBeenCalledTimes(1);
      });
    });

    describe('providerOptions', () => {
      it('should override base model providerOptions with retry model providerOptions', async () => {
        // Arrange
        const baseModel = MockLanguageModel.from({
          doGenerate: retryableError,
        });
        const fallbackModel = MockLanguageModel.from({
          doGenerate: mockResult,
        });
        const originalProviderOptions = { openai: { user: 'original-user' } };
        const retryProviderOptions = { openai: { user: 'retry-user' } };

        // Act
        await generateText({
          model: createRetryableModel({
            model: baseModel,
            retries: [
              {
                model: fallbackModel,
                providerOptions: retryProviderOptions,
              },
            ],
          }),
          prompt: 'Hello!',
          providerOptions: originalProviderOptions,
        });

        // Assert
        expect(baseModel.doGenerate).toHaveBeenCalledTimes(1);
        expect(fallbackModel.doGenerate).toHaveBeenCalledTimes(1);
        expect(baseModel.doGenerate).toHaveBeenCalledWith(
          expect.objectContaining({
            providerOptions: originalProviderOptions,
          }),
        );
        expect(fallbackModel.doGenerate).toHaveBeenCalledWith(
          expect.objectContaining({
            providerOptions: retryProviderOptions,
          }),
        );
      });
    });

    describe('timeout', () => {
      it('should create fresh abort signal with specified timeout on retry', async () => {
        // Arrange
        let baseModelSignal: AbortSignal | undefined;
        let fallbackModelSignal: AbortSignal | undefined;
        // Use TimeoutError (from AbortSignal.timeout()) which should be retried,
        // as opposed to AbortError (from user cancellation) which should NOT be retried
        const timeoutError = Errors.timeout();

        const baseModel = MockLanguageModel.from({
          doGenerate: async (opts: LanguageModelCallOptions) => {
            baseModelSignal = opts.abortSignal;
            throw timeoutError;
          },
        });

        const fallbackModel = MockLanguageModel.from({
          doGenerate: async (opts: LanguageModelCallOptions) => {
            fallbackModelSignal = opts.abortSignal;
            // Verify the new signal is not aborted
            if (opts.abortSignal?.aborted) {
              throw new Error('Should not be aborted with fresh signal');
            }
            return mockResult;
          },
        });

        // Create an already-aborted signal (simulates timeout that already fired)
        const controller = new AbortController();
        controller.abort(Errors.timeout());

        // Act
        await generateText({
          model: createRetryableModel({
            model: baseModel,
            retries: [
              {
                model: fallbackModel,
                timeout: 30000, // 30 second timeout for retry
              },
            ],
          }),
          prompt: 'Hello!',
          abortSignal: controller.signal,
        });

        // Assert
        expect(baseModel.doGenerate).toHaveBeenCalledTimes(1);
        expect(fallbackModel.doGenerate).toHaveBeenCalledTimes(1);

        // Base model should receive the original aborted signal
        expect(baseModelSignal?.aborted).toBe(true);

        // Fallback model should receive a fresh, non-aborted signal
        expect(fallbackModelSignal).toBeDefined();
        expect(fallbackModelSignal?.aborted).toBe(false);
        expect(baseModelSignal).not.toBe(fallbackModelSignal);
      });

      it('should not retry when base signal is aborted and retry has no timeout', async () => {
        // Arrange — simulates a framework-level abort (e.g. AI SDK
        // step/chunk timeout) where the inbound signal is already dead by
        // the time we catch the error. Without a fresh deadline on the
        // retry, the fallback would die instantly with the same abort.
        const abortError = Object.assign(
          new Error('The operation was aborted'),
          { name: 'AbortError' },
        );

        const baseModel = MockLanguageModel.from({
          doGenerate: async () => {
            throw abortError;
          },
        });
        const fallbackModel = MockLanguageModel.from({
          doGenerate: mockResult,
        });

        const onError = vi.fn();
        const onRetry = vi.fn();

        const controller = new AbortController();
        controller.abort();

        // Act
        const result = generateText({
          model: createRetryableModel({
            model: baseModel,
            retries: [fallbackModel],
            onError,
            onRetry,
          }),
          prompt: 'Hello!',
          abortSignal: controller.signal,
        });

        // Assert
        await expect(result).rejects.toThrow();
        expect(baseModel.doGenerate).toHaveBeenCalledTimes(1);
        expect(fallbackModel.doGenerate).toHaveBeenCalledTimes(0);
        expect(onError).toHaveBeenCalledTimes(1);
        expect(onRetry).toHaveBeenCalledTimes(0);
      });
    });

    describe('prompt', () => {
      it('should override prompt on retry', async () => {
        // Arrange
        const baseModel = MockLanguageModel.from({
          doGenerate: retryableError,
        });
        const fallbackModel = MockLanguageModel.from({
          doGenerate: mockResult,
        });
        const overridePrompt = [
          {
            role: 'user' as const,
            content: [{ type: 'text' as const, text: 'Modified prompt' }],
          },
        ];

        // Act
        await generateText({
          model: createRetryableModel({
            model: baseModel,
            retries: [
              { model: fallbackModel, options: { prompt: overridePrompt } },
            ],
          }),
          prompt: 'Original prompt',
        });

        // Assert
        expect(fallbackModel.doGenerate).toHaveBeenCalledWith(
          expect.objectContaining({ prompt: overridePrompt }),
        );
      });
    });

    describe('temperature', () => {
      it('should override temperature on retry', async () => {
        // Arrange
        const baseModel = MockLanguageModel.from({
          doGenerate: retryableError,
        });
        const fallbackModel = MockLanguageModel.from({
          doGenerate: mockResult,
        });

        // Act
        await generateText({
          model: createRetryableModel({
            model: baseModel,
            retries: [{ model: fallbackModel, options: { temperature: 0.5 } }],
          }),
          prompt: 'Hello!',
          temperature: 1.0,
        });

        // Assert
        expect(baseModel.doGenerate).toHaveBeenCalledWith(
          expect.objectContaining({ temperature: 1.0 }),
        );
        expect(fallbackModel.doGenerate).toHaveBeenCalledWith(
          expect.objectContaining({ temperature: 0.5 }),
        );
      });
    });

    describe('topP', () => {
      it('should override topP on retry', async () => {
        // Arrange
        const baseModel = MockLanguageModel.from({
          doGenerate: retryableError,
        });
        const fallbackModel = MockLanguageModel.from({
          doGenerate: mockResult,
        });

        // Act
        await generateText({
          model: createRetryableModel({
            model: baseModel,
            retries: [{ model: fallbackModel, options: { topP: 0.8 } }],
          }),
          prompt: 'Hello!',
          topP: 1.0,
        });

        // Assert
        expect(baseModel.doGenerate).toHaveBeenCalledWith(
          expect.objectContaining({ topP: 1.0 }),
        );
        expect(fallbackModel.doGenerate).toHaveBeenCalledWith(
          expect.objectContaining({ topP: 0.8 }),
        );
      });
    });

    describe('topK', () => {
      it('should override topK on retry', async () => {
        // Arrange
        const baseModel = MockLanguageModel.from({
          doGenerate: retryableError,
        });
        const fallbackModel = MockLanguageModel.from({
          doGenerate: mockResult,
        });

        // Act
        await generateText({
          model: createRetryableModel({
            model: baseModel,
            retries: [{ model: fallbackModel, options: { topK: 10 } }],
          }),
          prompt: 'Hello!',
          topK: 50,
        });

        // Assert
        expect(baseModel.doGenerate).toHaveBeenCalledWith(
          expect.objectContaining({ topK: 50 }),
        );
        expect(fallbackModel.doGenerate).toHaveBeenCalledWith(
          expect.objectContaining({ topK: 10 }),
        );
      });
    });

    describe('maxOutputTokens', () => {
      it('should override maxOutputTokens on retry', async () => {
        // Arrange
        const baseModel = MockLanguageModel.from({
          doGenerate: retryableError,
        });
        const fallbackModel = MockLanguageModel.from({
          doGenerate: mockResult,
        });

        // Act
        await generateText({
          model: createRetryableModel({
            model: baseModel,
            retries: [
              { model: fallbackModel, options: { maxOutputTokens: 500 } },
            ],
          }),
          prompt: 'Hello!',
          maxOutputTokens: 1000,
        });

        // Assert
        expect(baseModel.doGenerate).toHaveBeenCalledWith(
          expect.objectContaining({ maxOutputTokens: 1000 }),
        );
        expect(fallbackModel.doGenerate).toHaveBeenCalledWith(
          expect.objectContaining({ maxOutputTokens: 500 }),
        );
      });
    });

    describe('seed', () => {
      it('should override seed on retry', async () => {
        // Arrange
        const baseModel = MockLanguageModel.from({
          doGenerate: retryableError,
        });
        const fallbackModel = MockLanguageModel.from({
          doGenerate: mockResult,
        });

        // Act
        await generateText({
          model: createRetryableModel({
            model: baseModel,
            retries: [{ model: fallbackModel, options: { seed: 42 } }],
          }),
          prompt: 'Hello!',
          seed: 123,
        });

        // Assert
        expect(baseModel.doGenerate).toHaveBeenCalledWith(
          expect.objectContaining({ seed: 123 }),
        );
        expect(fallbackModel.doGenerate).toHaveBeenCalledWith(
          expect.objectContaining({ seed: 42 }),
        );
      });
    });

    describe('stopSequences', () => {
      it('should override stopSequences on retry', async () => {
        // Arrange
        const baseModel = MockLanguageModel.from({
          doGenerate: retryableError,
        });
        const fallbackModel = MockLanguageModel.from({
          doGenerate: mockResult,
        });

        // Act
        await generateText({
          model: createRetryableModel({
            model: baseModel,
            retries: [
              {
                model: fallbackModel,
                options: { stopSequences: ['RETRY_STOP'] },
              },
            ],
          }),
          prompt: 'Hello!',
          stopSequences: ['ORIGINAL_STOP'],
        });

        // Assert
        expect(baseModel.doGenerate).toHaveBeenCalledWith(
          expect.objectContaining({ stopSequences: ['ORIGINAL_STOP'] }),
        );
        expect(fallbackModel.doGenerate).toHaveBeenCalledWith(
          expect.objectContaining({ stopSequences: ['RETRY_STOP'] }),
        );
      });
    });

    describe('presencePenalty', () => {
      it('should override presencePenalty on retry', async () => {
        // Arrange
        const baseModel = MockLanguageModel.from({
          doGenerate: retryableError,
        });
        const fallbackModel = MockLanguageModel.from({
          doGenerate: mockResult,
        });

        // Act
        await generateText({
          model: createRetryableModel({
            model: baseModel,
            retries: [
              { model: fallbackModel, options: { presencePenalty: 0.5 } },
            ],
          }),
          prompt: 'Hello!',
          presencePenalty: 0.0,
        });

        // Assert
        expect(baseModel.doGenerate).toHaveBeenCalledWith(
          expect.objectContaining({ presencePenalty: 0.0 }),
        );
        expect(fallbackModel.doGenerate).toHaveBeenCalledWith(
          expect.objectContaining({ presencePenalty: 0.5 }),
        );
      });
    });

    describe('frequencyPenalty', () => {
      it('should override frequencyPenalty on retry', async () => {
        // Arrange
        const baseModel = MockLanguageModel.from({
          doGenerate: retryableError,
        });
        const fallbackModel = MockLanguageModel.from({
          doGenerate: mockResult,
        });

        // Act
        await generateText({
          model: createRetryableModel({
            model: baseModel,
            retries: [
              { model: fallbackModel, options: { frequencyPenalty: 0.8 } },
            ],
          }),
          prompt: 'Hello!',
          frequencyPenalty: 0.2,
        });

        // Assert
        expect(baseModel.doGenerate).toHaveBeenCalledWith(
          expect.objectContaining({ frequencyPenalty: 0.2 }),
        );
        expect(fallbackModel.doGenerate).toHaveBeenCalledWith(
          expect.objectContaining({ frequencyPenalty: 0.8 }),
        );
      });
    });

    describe('headers', () => {
      it('should override headers on retry', async () => {
        // Arrange
        const baseModel = MockLanguageModel.from({
          doGenerate: retryableError,
        });
        const fallbackModel = MockLanguageModel.from({
          doGenerate: mockResult,
        });
        // Use lowercase headers to match what AI SDK expects
        const retryHeaders = { 'x-retry': 'retry' };

        // Act
        await generateText({
          model: createRetryableModel({
            model: baseModel,
            retries: [
              { model: fallbackModel, options: { headers: retryHeaders } },
            ],
          }),
          prompt: 'Hello!',
        });

        // Assert - check that retry headers are passed to fallback model
        expect(fallbackModel.doGenerate).toHaveBeenCalledWith(
          expect.objectContaining({
            headers: expect.objectContaining({
              'x-retry': 'retry',
            }),
          }),
        );
      });
    });
  });

  describe('RetryError', () => {
    it('should throw RetryError when all retry attempts are exhausted', async () => {
      // Arrange
      const baseModel = MockLanguageModel.from({ doGenerate: retryableError });
      const fallbackModel1 = MockLanguageModel.from({
        doGenerate: nonRetryableError,
      });
      const fallbackModel2 = MockLanguageModel.from({
        doGenerate: retryableError,
      });

      // Act & Assert
      try {
        await generateText({
          model: createRetryableModel({
            model: baseModel,
            retries: [fallbackModel1, fallbackModel2],
          }),
          prompt: 'Hello!',
        });
        expect.unreachable(
          'Should throw RetryError when all attempts are exhausted',
        );
      } catch (error) {
        expect(error).toBeInstanceOf(RetryError);

        const retryError = error as RetryError;
        expect(retryError.reason).toBe('maxRetriesExceeded');
        expect(retryError.errors).toHaveLength(3);
        expect(retryError.errors[0]).toBe(retryableError);
        expect(retryError.errors[1]).toBe(nonRetryableError);
        expect(retryError.errors[2]).toBe(retryableError);
      }
    });

    it('should throw original error directly on first attempt with no retryables', async () => {
      // Arrange
      const baseModel = MockLanguageModel.from({ doGenerate: retryableError });

      // Act & Assert
      try {
        await generateText({
          model: createRetryableModel({
            model: baseModel,
            retries: [], // No retry models
          }),
          prompt: 'Hello!',
          maxRetries: 0, // No automatic retries
        });
        expect.unreachable(
          'Should throw original error on first attempt with no retries',
        );
      } catch (error) {
        expect(error).not.toBeInstanceOf(RetryError);
        expect(error).toBe(retryableError);
      }
    });

    it('should throw original error directly when retryable returns undefined', async () => {
      // Arrange
      const baseModel = MockLanguageModel.from({
        doGenerate: nonRetryableError,
      });

      // Act & Assert
      try {
        await generateText({
          model: createRetryableModel({
            model: baseModel,
            retries: [() => undefined],
          }),
          prompt: 'Hello!',
          maxRetries: 0, // No automatic retries
        });
        expect.unreachable(
          'Should throw original error when retry models return undefined',
        );
      } catch (error) {
        expect(error).not.toBeInstanceOf(RetryError);
        expect(error).toBe(nonRetryableError);
      }
    });
  });

  describe(`reset`, () => {
    describe(`after-request (default)`, () => {
      it(`should reset to base model on every request`, async () => {
        // Arrange
        let callCount = 0;
        const baseModel = MockLanguageModel.from({
          doGenerate: async () => {
            callCount++;
            if (callCount <= 1) throw retryableError;
            return mockResult;
          },
        });
        const fallbackModel = MockLanguageModel.from({
          doGenerate: mockResult,
        });

        const retryableModel = createRetryableModel({
          model: baseModel,
          retries: [{ model: fallbackModel, maxAttempts: 1 }],
        });

        // Act — first request: base fails, fallback succeeds
        const result1 = await generateText({
          model: retryableModel,
          prompt,
        });

        // Act — second request: base model is used again (reset), succeeds this time
        const result2 = await generateText({
          model: retryableModel,
          prompt,
        });

        // Assert
        expect(result1.text).toBe(mockResultText);
        expect(result2.text).toBe(mockResultText);
        expect(baseModel.doGenerate).toHaveBeenCalledTimes(2);
        expect(fallbackModel.doGenerate).toHaveBeenCalledTimes(1);
      });
    });

    describe(`after-N-requests`, () => {
      it(`should use sticky model for N subsequent requests then reset`, async () => {
        // Arrange
        const baseModel = MockLanguageModel.from({
          doGenerate: retryableError,
        });
        const fallbackModel = MockLanguageModel.from({
          doGenerate: mockResult,
        });

        const retryableModel = createRetryableModel({
          model: baseModel,
          retries: [{ model: fallbackModel, maxAttempts: 1 }],
          reset: `after-2-requests`,
        });

        // Act — request 1: base fails, fallback succeeds → sticky set
        const result1 = await generateText({ model: retryableModel, prompt });

        // Act — request 2: sticky (fallback) used directly
        const result2 = await generateText({ model: retryableModel, prompt });

        // Act — request 3: sticky (fallback) used directly (last sticky request)
        const result3 = await generateText({ model: retryableModel, prompt });

        // Act — request 4: sticky expired, back to base model → base fails, fallback retried
        const result4 = await generateText({ model: retryableModel, prompt });

        // Assert
        expect(result1.text).toBe(mockResultText);
        expect(result2.text).toBe(mockResultText);
        expect(result3.text).toBe(mockResultText);
        expect(result4.text).toBe(mockResultText);
        // Request 1: base(1) + fallback(1), Request 2: fallback(1), Request 3: fallback(1),
        // Request 4: base(1) + fallback(1)
        expect(baseModel.doGenerate).toHaveBeenCalledTimes(2);
        expect(fallbackModel.doGenerate).toHaveBeenCalledTimes(4);
      });

      it(`should not set sticky model when no retry occurred`, async () => {
        // Arrange
        const baseModel = MockLanguageModel.from({ doGenerate: mockResult });
        const fallbackModel = MockLanguageModel.from({
          doGenerate: mockResult,
        });

        const retryableModel = createRetryableModel({
          model: baseModel,
          retries: [{ model: fallbackModel, maxAttempts: 1 }],
          reset: `after-3-requests`,
        });

        // Act
        await generateText({ model: retryableModel, prompt });
        await generateText({ model: retryableModel, prompt });

        // Assert — base model used both times, no fallback
        expect(baseModel.doGenerate).toHaveBeenCalledTimes(2);
        expect(fallbackModel.doGenerate).toHaveBeenCalledTimes(0);
      });

      it(`should reset counter when a new sticky model is set`, async () => {
        // Arrange
        let fallback1CallCount = 0;
        const baseModel = MockLanguageModel.from({
          doGenerate: retryableError,
        });
        const fallbackModel1 = MockLanguageModel.from({
          doGenerate: async () => {
            fallback1CallCount++;
            /** Fail on second use (when used as sticky on request 2) */
            if (fallback1CallCount >= 2) throw retryableError;
            return mockResult;
          },
        });
        const fallbackModel2 = MockLanguageModel.from({
          doGenerate: mockResult,
        });

        const retryableModel = createRetryableModel({
          model: baseModel,
          retries: [
            { model: fallbackModel1, maxAttempts: 1 },
            { model: fallbackModel2, maxAttempts: 1 },
          ],
          reset: `after-2-requests`,
        });

        // Act — request 1: base fails → fallback1 succeeds → sticky = fallback1
        await generateText({ model: retryableModel, prompt });

        // Act — request 2: sticky (fallback1) fails → fallback2 succeeds → sticky = fallback2, counter resets to 2
        await generateText({ model: retryableModel, prompt });

        // Act — request 3: sticky (fallback2) used directly (remaining = 1)
        await generateText({ model: retryableModel, prompt });

        // Act — request 4: sticky (fallback2) used directly (remaining = 0)
        await generateText({ model: retryableModel, prompt });

        // Assert
        expect(baseModel.doGenerate).toHaveBeenCalledTimes(1);
        expect(fallbackModel1.doGenerate).toHaveBeenCalledTimes(2);
        expect(fallbackModel2.doGenerate).toHaveBeenCalledTimes(3);
      });
    });

    describe(`after-N-seconds`, () => {
      it(`should use sticky model within time window then reset`, async () => {
        // Arrange
        vi.useFakeTimers();

        const baseModel = MockLanguageModel.from({
          doGenerate: retryableError,
        });
        const fallbackModel = MockLanguageModel.from({
          doGenerate: mockResult,
        });

        const retryableModel = createRetryableModel({
          model: baseModel,
          retries: [{ model: fallbackModel, maxAttempts: 1 }],
          reset: `after-5-seconds`,
        });

        // Act — request 1: base fails, fallback succeeds → sticky set
        await generateText({ model: retryableModel, prompt });

        // Act — request 2: within 5s, sticky (fallback) used directly
        vi.advanceTimersByTime(2_000);
        await generateText({ model: retryableModel, prompt });

        // Act — request 3: advance past 5s, sticky expired → back to base → fails → fallback
        vi.advanceTimersByTime(4_000);
        await generateText({ model: retryableModel, prompt });

        // Assert
        expect(baseModel.doGenerate).toHaveBeenCalledTimes(2);
        expect(fallbackModel.doGenerate).toHaveBeenCalledTimes(3);

        vi.useRealTimers();
      });
    });
  });
});

describe('streamText', () => {
  it('should stream successfully when no errors occur', async () => {
    const baseModel = MockLanguageModel.from({
      doStream: mockStreamChunks,
    });

    const retryableModel = createRetryableModel({
      model: baseModel,
      retries: [],
    });

    const result = streamText({
      model: retryableModel,
      prompt,
    });

    const chunks = await Iterables.toArray(result.fullStream);

    expect(baseModel.doStream).toHaveBeenCalledTimes(1);
    expect(chunksToText(chunks)).toBe('Hello, world!');
  });

  describe('retries', () => {
    describe('error-based retries', () => {
      it('should retry when error occurs at stream creation', async () => {
        const baseModel = MockLanguageModel.from({ doStream: retryableError });

        const fallbackModel = MockLanguageModel.from({
          doStream: mockStreamChunks,
        });

        const retryableModel = createRetryableModel({
          model: baseModel,
          retries: [fallbackModel],
        });

        const result = streamText({
          model: retryableModel,
          prompt,
        });

        const chunks = await Iterables.toArray(result.fullStream);

        expect(baseModel.doStream).toHaveBeenCalledTimes(1);
        expect(fallbackModel.doStream).toHaveBeenCalledTimes(1);
        expect(chunks).toMatchInlineSnapshot(`
          [
            {
              "type": "start",
            },
            {
              "request": {
                "body": undefined,
                "messages": undefined,
              },
              "type": "start-step",
              "warnings": [],
            },
            {
              "id": "1",
              "type": "text-start",
            },
            {
              "id": "1",
              "providerMetadata": undefined,
              "text": "Hello",
              "type": "text-delta",
            },
            {
              "id": "1",
              "providerMetadata": undefined,
              "text": ", ",
              "type": "text-delta",
            },
            {
              "id": "1",
              "providerMetadata": undefined,
              "text": "world!",
              "type": "text-delta",
            },
            {
              "id": "1",
              "type": "text-end",
            },
            {
              "finishReason": "stop",
              "providerMetadata": {
                "testProvider": {
                  "testKey": "testValue",
                },
              },
              "rawFinishReason": "stop",
              "response": {
                "headers": undefined,
                "id": "id-0",
                "modelId": "mock-model-id",
                "timestamp": 1970-01-01T00:00:00.000Z,
              },
              "type": "finish-step",
              "usage": {
                "inputTokenDetails": {
                  "cacheReadTokens": 0,
                  "cacheWriteTokens": 0,
                  "noCacheTokens": 10,
                },
                "inputTokens": 10,
                "outputTokenDetails": {
                  "reasoningTokens": 0,
                  "textTokens": 20,
                },
                "outputTokens": 20,
                "raw": undefined,
                "totalTokens": 30,
              },
            },
            {
              "finishReason": "stop",
              "rawFinishReason": "stop",
              "totalUsage": {
                "inputTokenDetails": {
                  "cacheReadTokens": 0,
                  "cacheWriteTokens": 0,
                  "noCacheTokens": 10,
                },
                "inputTokens": 10,
                "outputTokenDetails": {
                  "reasoningTokens": 0,
                  "textTokens": 20,
                },
                "outputTokens": 20,
                "totalTokens": 30,
              },
              "type": "finish",
            },
          ]
        `);
      });

      it('should retry when error occurs at the stream start', async () => {
        const baseModel = MockLanguageModel.from({
          doStream: [
            { type: 'stream-start', warnings: [] },
            {
              type: 'error',
              error: { type: 'overloaded_error', message: 'Overloaded' },
            },
          ],
        });

        const fallbackModel = MockLanguageModel.from({
          doStream: mockStreamChunks,
        });

        const retryableModel = createRetryableModel({
          model: baseModel,
          retries: [fallbackModel],
        });

        const result = streamText({
          model: retryableModel,
          prompt,
        });

        const chunks = await Iterables.toArray(result.fullStream);

        expect(baseModel.doStream).toHaveBeenCalledTimes(1);
        expect(fallbackModel.doStream).toHaveBeenCalledTimes(1);
        expect(chunks).toMatchInlineSnapshot(`
          [
            {
              "type": "start",
            },
            {
              "request": {
                "body": undefined,
                "messages": undefined,
              },
              "type": "start-step",
              "warnings": [],
            },
            {
              "id": "1",
              "type": "text-start",
            },
            {
              "id": "1",
              "providerMetadata": undefined,
              "text": "Hello",
              "type": "text-delta",
            },
            {
              "id": "1",
              "providerMetadata": undefined,
              "text": ", ",
              "type": "text-delta",
            },
            {
              "id": "1",
              "providerMetadata": undefined,
              "text": "world!",
              "type": "text-delta",
            },
            {
              "id": "1",
              "type": "text-end",
            },
            {
              "finishReason": "stop",
              "providerMetadata": {
                "testProvider": {
                  "testKey": "testValue",
                },
              },
              "rawFinishReason": "stop",
              "response": {
                "headers": undefined,
                "id": "id-0",
                "modelId": "mock-model-id",
                "timestamp": 1970-01-01T00:00:00.000Z,
              },
              "type": "finish-step",
              "usage": {
                "inputTokenDetails": {
                  "cacheReadTokens": 0,
                  "cacheWriteTokens": 0,
                  "noCacheTokens": 10,
                },
                "inputTokens": 10,
                "outputTokenDetails": {
                  "reasoningTokens": 0,
                  "textTokens": 20,
                },
                "outputTokens": 20,
                "raw": undefined,
                "totalTokens": 30,
              },
            },
            {
              "finishReason": "stop",
              "rawFinishReason": "stop",
              "totalUsage": {
                "inputTokenDetails": {
                  "cacheReadTokens": 0,
                  "cacheWriteTokens": 0,
                  "noCacheTokens": 10,
                },
                "inputTokens": 10,
                "outputTokenDetails": {
                  "reasoningTokens": 0,
                  "textTokens": 20,
                },
                "outputTokens": 20,
                "totalTokens": 30,
              },
              "type": "finish",
            },
          ]
        `);
      });

      it('should retry when consective errors occur', async () => {
        const baseModel = MockLanguageModel.from({
          doStream: [
            { type: 'stream-start', warnings: [] },
            {
              type: 'error',
              error: { type: 'overloaded_error', message: 'Overloaded' },
            },
          ],
        });

        const fallbackModel1 = MockLanguageModel.from({
          doStream: retryableError,
        });
        const fallbackModel2 = MockLanguageModel.from({
          doStream: mockStreamChunks,
        });

        const retryableModel = createRetryableModel({
          model: baseModel,
          retries: [fallbackModel1, fallbackModel2],
        });

        const result = streamText({
          model: retryableModel,
          prompt,
        });

        const chunks = await Iterables.toArray(result.fullStream);

        expect(baseModel.doStream).toHaveBeenCalledTimes(1);
        expect(fallbackModel1.doStream).toHaveBeenCalledTimes(1);
        expect(fallbackModel2.doStream).toHaveBeenCalledTimes(1);
        expect(chunks).toMatchInlineSnapshot(`
          [
            {
              "type": "start",
            },
            {
              "request": {
                "body": undefined,
                "messages": undefined,
              },
              "type": "start-step",
              "warnings": [],
            },
            {
              "id": "1",
              "type": "text-start",
            },
            {
              "id": "1",
              "providerMetadata": undefined,
              "text": "Hello",
              "type": "text-delta",
            },
            {
              "id": "1",
              "providerMetadata": undefined,
              "text": ", ",
              "type": "text-delta",
            },
            {
              "id": "1",
              "providerMetadata": undefined,
              "text": "world!",
              "type": "text-delta",
            },
            {
              "id": "1",
              "type": "text-end",
            },
            {
              "finishReason": "stop",
              "providerMetadata": {
                "testProvider": {
                  "testKey": "testValue",
                },
              },
              "rawFinishReason": "stop",
              "response": {
                "headers": undefined,
                "id": "id-0",
                "modelId": "mock-model-id",
                "timestamp": 1970-01-01T00:00:00.000Z,
              },
              "type": "finish-step",
              "usage": {
                "inputTokenDetails": {
                  "cacheReadTokens": 0,
                  "cacheWriteTokens": 0,
                  "noCacheTokens": 10,
                },
                "inputTokens": 10,
                "outputTokenDetails": {
                  "reasoningTokens": 0,
                  "textTokens": 20,
                },
                "outputTokens": 20,
                "raw": undefined,
                "totalTokens": 30,
              },
            },
            {
              "finishReason": "stop",
              "rawFinishReason": "stop",
              "totalUsage": {
                "inputTokenDetails": {
                  "cacheReadTokens": 0,
                  "cacheWriteTokens": 0,
                  "noCacheTokens": 10,
                },
                "inputTokens": 10,
                "outputTokenDetails": {
                  "reasoningTokens": 0,
                  "textTokens": 20,
                },
                "outputTokens": 20,
                "totalTokens": 30,
              },
              "type": "finish",
            },
          ]
        `);
      });

      it('should NOT retry when error occurs during streaming', async () => {
        vi.useFakeTimers();
        vi.setSystemTime(0);

        const baseModel = MockLanguageModel.from({
          doStream: [
            Language.streamStart(),
            { type: 'text-start', id: '1' },
            { type: 'text-delta', id: '1', delta: 'Hello' },
            Language.streamError(new Error('Overloaded')),
          ],
        });

        const fallbackModel = MockLanguageModel.from({
          doStream: mockStreamChunks,
        });

        const retryableModel = createRetryableModel({
          model: baseModel,
          retries: [fallbackModel],
        });

        const result = streamText({
          model: retryableModel,
          prompt,
          ...Options.stream,
        });

        const chunks = await Iterables.toArray(result.fullStream);

        expect(baseModel.doStream).toHaveBeenCalledTimes(1);
        expect(fallbackModel.doStream).toHaveBeenCalledTimes(0);
        expect(chunks).toMatchInlineSnapshot(`
          [
            {
              "type": "start",
            },
            {
              "request": {
                "body": undefined,
                "messages": undefined,
              },
              "type": "start-step",
              "warnings": [],
            },
            {
              "id": "1",
              "type": "text-start",
            },
            {
              "id": "1",
              "providerMetadata": undefined,
              "text": "Hello",
              "type": "text-delta",
            },
            {
              "error": [Error: Overloaded],
              "type": "error",
            },
            {
              "finishReason": "error",
              "providerMetadata": undefined,
              "rawFinishReason": undefined,
              "response": {
                "headers": undefined,
                "id": "aitxt-mock-id",
                "modelId": "mock-model-148",
                "timestamp": 1970-01-01T00:00:00.000Z,
              },
              "type": "finish-step",
              "usage": {
                "inputTokenDetails": {
                  "cacheReadTokens": undefined,
                  "cacheWriteTokens": undefined,
                  "noCacheTokens": undefined,
                },
                "inputTokens": undefined,
                "outputTokenDetails": {
                  "reasoningTokens": undefined,
                  "textTokens": undefined,
                },
                "outputTokens": undefined,
                "raw": undefined,
                "totalTokens": undefined,
              },
            },
            {
              "finishReason": "error",
              "rawFinishReason": undefined,
              "totalUsage": {
                "inputTokenDetails": {
                  "cacheReadTokens": undefined,
                  "cacheWriteTokens": undefined,
                  "noCacheTokens": undefined,
                },
                "inputTokens": undefined,
                "outputTokenDetails": {
                  "reasoningTokens": undefined,
                  "textTokens": undefined,
                },
                "outputTokens": undefined,
                "totalTokens": undefined,
              },
              "type": "finish",
            },
          ]
        `);

        vi.useRealTimers();
      });

      it('should propagate error as stream part when no retryable matches at stream start', async () => {
        // Arrange
        vi.useFakeTimers();
        vi.setSystemTime(0);

        const error = new Error('Overloaded');

        const baseModel = MockLanguageModel.from({
          doStream: [Language.streamStart(), Language.streamError(error)],
        });

        const noopRetryable: ModelRetryable<LanguageModel> = () => undefined;

        const retryableModel = createRetryableModel({
          model: baseModel,
          retries: [noopRetryable],
        });

        const onErrorSpy = vi.fn();

        // Act
        const result = streamText({
          model: retryableModel,
          prompt,
          onError: onErrorSpy,
          ...Options.stream,
        });

        const chunks = await Iterables.toArray(result.fullStream);

        // Assert
        expect(baseModel.doStream).toHaveBeenCalledTimes(1);

        const errorChunk = chunks.find((c) => c.type === 'error');
        expect(errorChunk).toEqual({ type: 'error', error });

        expect(onErrorSpy).toHaveBeenCalledTimes(1);
        const onErrorArg = onErrorSpy.mock.calls[0]![0];
        expect(onErrorArg).toEqual({ error });

        expect(chunks).toMatchInlineSnapshot(`
          [
            {
              "type": "start",
            },
            {
              "request": {
                "body": undefined,
                "messages": undefined,
              },
              "type": "start-step",
              "warnings": [],
            },
            {
              "error": [Error: Overloaded],
              "type": "error",
            },
            {
              "finishReason": "error",
              "providerMetadata": undefined,
              "rawFinishReason": undefined,
              "response": {
                "headers": undefined,
                "id": "aitxt-mock-id",
                "modelId": "mock-model-150",
                "timestamp": 1970-01-01T00:00:00.000Z,
              },
              "type": "finish-step",
              "usage": {
                "inputTokenDetails": {
                  "cacheReadTokens": undefined,
                  "cacheWriteTokens": undefined,
                  "noCacheTokens": undefined,
                },
                "inputTokens": undefined,
                "outputTokenDetails": {
                  "reasoningTokens": undefined,
                  "textTokens": undefined,
                },
                "outputTokens": undefined,
                "raw": undefined,
                "totalTokens": undefined,
              },
            },
            {
              "finishReason": "error",
              "rawFinishReason": undefined,
              "totalUsage": {
                "inputTokenDetails": {
                  "cacheReadTokens": undefined,
                  "cacheWriteTokens": undefined,
                  "noCacheTokens": undefined,
                },
                "inputTokens": undefined,
                "outputTokenDetails": {
                  "reasoningTokens": undefined,
                  "textTokens": undefined,
                },
                "outputTokens": undefined,
                "totalTokens": undefined,
              },
              "type": "finish",
            },
          ]
        `);

        vi.useRealTimers();
      });

      describe('mid-stream errors before first content', () => {
        /**
         * A stream that emits `stream-start` and then errors the stream
         * itself via `controller.error` (the real-world body-stall / undici
         * `bodyTimeout` signature), rather than emitting an `error` part.
         */
        const streamStartThenError = (error: unknown) =>
          new ReadableStream<LanguageModelStreamPart>({
            start(controller) {
              controller.enqueue({ type: 'stream-start', warnings: [] });
              controller.error(error);
            },
          });

        it('should retry when the stream errors before any content', async () => {
          // Arrange
          const error = new Error('body timeout');
          const baseModel = MockLanguageModel.from({
            doStream: streamStartThenError(error),
          });
          const fallbackModel = MockLanguageModel.from({
            doStream: successStreamChunks('Recovered'),
          });
          const retryableModel = createRetryableModel({
            model: baseModel,
            retries: [fallbackModel],
          });

          // Act
          const { stream } = await retryableModel.doStream(
            MockLanguageModel.callOptions(),
          );
          const parts = await Streams.toArray(stream);

          // Assert
          expect(baseModel.doStream).toHaveBeenCalledTimes(1);
          expect(fallbackModel.doStream).toHaveBeenCalledTimes(1);
          expect(partsToText(parts)).toBe('Recovered');
        });

        it('should emit exactly one stream-start when retrying before content', async () => {
          // Arrange
          const error = new Error('Overloaded');
          const baseModel = MockLanguageModel.from({
            doStream: [
              { type: 'stream-start', warnings: [] },
              { type: 'error', error },
            ],
          });
          const fallbackModel = MockLanguageModel.from({
            doStream: successStreamChunks('Recovered'),
          });
          const retryableModel = createRetryableModel({
            model: baseModel,
            retries: [fallbackModel],
          });

          // Act
          const { stream } = await retryableModel.doStream(
            MockLanguageModel.callOptions(),
          );
          const parts = await Streams.toArray(stream);

          // Assert
          const startCount = parts.filter(
            (p) => p.type === 'stream-start',
          ).length;
          expect(startCount).toBe(1);
        });

        it('should emit the fallback preamble, not the primary preamble', async () => {
          // Arrange
          const error = new Error('Overloaded');
          const baseModel = MockLanguageModel.from({
            doStream: [
              {
                type: 'stream-start',
                warnings: [{ type: 'other', message: 'primary-warning' }],
              },
              {
                type: 'response-metadata',
                id: 'primary-id',
                modelId: 'primary-model',
                timestamp: new Date(0),
              },
              { type: 'error', error },
            ],
          });
          const fallbackModel = MockLanguageModel.from({
            doStream: [
              Language.streamStart(),
              Language.streamResponseMetadata({
                id: 'fallback-id',
                modelId: 'fallback-model',
                timestamp: new Date(0),
              }),
              ...Language.streamText('Recovered', { id: '1' }),
              Language.streamFinish(),
            ],
          });
          const retryableModel = createRetryableModel({
            model: baseModel,
            retries: [fallbackModel],
          });

          // Act
          const { stream } = await retryableModel.doStream(
            MockLanguageModel.callOptions(),
          );
          const parts = await Streams.toArray(stream);

          // Assert
          const startParts = parts.filter((p) => p.type === 'stream-start');
          const metadataParts = parts.filter(
            (p) => p.type === 'response-metadata',
          );
          expect(startParts.length).toBe(1);
          expect(startParts[0]).toEqual({ type: 'stream-start', warnings: [] });
          expect(metadataParts.length).toBe(1);
          expect(metadataParts[0]).toEqual({
            type: 'response-metadata',
            id: 'fallback-id',
            modelId: 'fallback-model',
            timestamp: new Date(0),
          });
        });

        it('should deliver fallback output through streamText when the stream errors before content', async () => {
          // Arrange
          const error = new Error('body timeout');
          const baseModel = MockLanguageModel.from({
            doStream: streamStartThenError(error),
          });
          const fallbackModel = MockLanguageModel.from({
            doStream: mockStreamChunks,
          });
          const retryableModel = createRetryableModel({
            model: baseModel,
            retries: [fallbackModel],
          });

          // Act
          const result = streamText({
            model: retryableModel,
            prompt,
            ...Options.stream,
          });
          const chunks = await Iterables.toArray(result.fullStream);

          // Assert
          expect(baseModel.doStream).toHaveBeenCalledTimes(1);
          expect(fallbackModel.doStream).toHaveBeenCalledTimes(1);
          expect(chunksToText(chunks)).toBe('Hello, world!');
          expect(errorFromChunks(chunks)).toBe(null);
        });
      });
    });

    describe('result-based retries', () => {
      it('should retry with results', async () => {
        // Arrange
        const baseModel = MockLanguageModel.from({
          doStream: contentFilterStreamChunks,
        });
        const fallbackModel = MockLanguageModel.from({
          doStream: mockStreamChunks,
        });
        const fallbackRetryable: ModelRetryable<LanguageModel> = (context) => {
          if (
            isResultAttempt(context.current) &&
            context.current.result.finishReason.unified === 'content-filter'
          ) {
            return { model: fallbackModel, maxAttempts: 1 };
          }
          return undefined;
        };

        // Act
        const result = streamText({
          model: createRetryableModel({
            model: baseModel,
            retries: [fallbackRetryable],
          }),
          prompt,
        });

        const chunks = await Iterables.toArray(result.fullStream);

        // Assert
        expect(baseModel.doStream).toHaveBeenCalledTimes(1);
        expect(fallbackModel.doStream).toHaveBeenCalledTimes(1);
        expect(finishReason(chunks)).not.toBe('content-filter');
        expect(finishReason(chunks)).toBe('stop');
        expect(chunks).toMatchInlineSnapshot(`
          [
            {
              "type": "start",
            },
            {
              "request": {
                "body": undefined,
                "messages": undefined,
              },
              "type": "start-step",
              "warnings": [],
            },
            {
              "id": "1",
              "type": "text-start",
            },
            {
              "id": "1",
              "providerMetadata": undefined,
              "text": "Hello",
              "type": "text-delta",
            },
            {
              "id": "1",
              "providerMetadata": undefined,
              "text": ", ",
              "type": "text-delta",
            },
            {
              "id": "1",
              "providerMetadata": undefined,
              "text": "world!",
              "type": "text-delta",
            },
            {
              "id": "1",
              "type": "text-end",
            },
            {
              "finishReason": "stop",
              "providerMetadata": {
                "testProvider": {
                  "testKey": "testValue",
                },
              },
              "rawFinishReason": "stop",
              "response": {
                "headers": undefined,
                "id": "id-0",
                "modelId": "mock-model-id",
                "timestamp": 1970-01-01T00:00:00.000Z,
              },
              "type": "finish-step",
              "usage": {
                "inputTokenDetails": {
                  "cacheReadTokens": 0,
                  "cacheWriteTokens": 0,
                  "noCacheTokens": 10,
                },
                "inputTokens": 10,
                "outputTokenDetails": {
                  "reasoningTokens": 0,
                  "textTokens": 20,
                },
                "outputTokens": 20,
                "raw": undefined,
                "totalTokens": 30,
              },
            },
            {
              "finishReason": "stop",
              "rawFinishReason": "stop",
              "totalUsage": {
                "inputTokenDetails": {
                  "cacheReadTokens": 0,
                  "cacheWriteTokens": 0,
                  "noCacheTokens": 10,
                },
                "inputTokens": 10,
                "outputTokenDetails": {
                  "reasoningTokens": 0,
                  "textTokens": 20,
                },
                "outputTokens": 20,
                "totalTokens": 30,
              },
              "type": "finish",
            },
          ]
        `);
      });

      describe('mid-stream errors after content', () => {
        /**
         * Content flows, then the stream itself errors via `controller.error`
         * (the real-world body-stall / undici `bodyTimeout` signature). A small
         * gap lets the content commit before the error, as a real socket does.
         */
        const contentThenError = (error: unknown) =>
          new ReadableStream<LanguageModelStreamPart>({
            async start(controller) {
              controller.enqueue({ type: 'stream-start', warnings: [] });
              controller.enqueue({ type: 'text-start', id: '1' });
              controller.enqueue({
                type: 'text-delta',
                id: '1',
                delta: 'Partial',
              });
              await new Promise((resolve) => setTimeout(resolve, 10));
              controller.error(error);
            },
          });

        it('should NOT retry when the stream errors after content (no duplication)', async () => {
          // Arrange
          const error = new Error('body timeout');
          const baseModel = MockLanguageModel.from({
            doStream: contentThenError(error),
          });
          const fallbackModel = MockLanguageModel.from({
            doStream: successStreamChunks('Recovered'),
          });
          const retryableModel = createRetryableModel({
            model: baseModel,
            retries: [fallbackModel],
          });

          // Act
          const { stream } = await retryableModel.doStream(
            MockLanguageModel.callOptions(),
          );
          const parts = await Streams.toArray(stream);

          // Assert — committed on 'Partial', so no fail-over and no duplication.
          expect(baseModel.doStream).toHaveBeenCalledTimes(1);
          expect(fallbackModel.doStream).toHaveBeenCalledTimes(0);
          expect(partsToText(parts)).toBe('Partial');
          expect(parts.at(-1)?.type).toBe('error');
        });
      });

      it('should NOT retry when content was already streamed before content-filter finish', async () => {
        // Arrange
        const baseModel = MockLanguageModel.from({
          doStream: contentFilterAfterContentStreamChunks,
        });
        const fallbackModel = MockLanguageModel.from({
          doStream: mockStreamChunks,
        });
        const fallbackRetryable: ModelRetryable<LanguageModel> = (context) => {
          if (
            isResultAttempt(context.current) &&
            context.current.result.finishReason.unified === 'content-filter'
          ) {
            return { model: fallbackModel, maxAttempts: 1 };
          }
          return undefined;
        };

        // Act
        const result = streamText({
          model: createRetryableModel({
            model: baseModel,
            retries: [fallbackRetryable],
          }),
          prompt,
        });

        const chunks = await Iterables.toArray(result.fullStream);

        // Assert
        expect(baseModel.doStream).toHaveBeenCalledTimes(1);
        expect(fallbackModel.doStream).toHaveBeenCalledTimes(0);
        expect(chunksToText(chunks)).toBe(refusalText);
        expect(finishReason(chunks)).toBe('content-filter');
        expect(chunks).toMatchInlineSnapshot(`
          [
            {
              "type": "start",
            },
            {
              "request": {
                "body": undefined,
                "messages": undefined,
              },
              "type": "start-step",
              "warnings": [],
            },
            {
              "id": "1",
              "type": "text-start",
            },
            {
              "id": "1",
              "providerMetadata": undefined,
              "text": "I'm sorry, but I cannot assist with that request.",
              "type": "text-delta",
            },
            {
              "id": "1",
              "type": "text-end",
            },
            {
              "finishReason": "content-filter",
              "providerMetadata": {
                "testProvider": {
                  "testKey": "testValue",
                },
              },
              "rawFinishReason": "content-filter",
              "response": {
                "headers": undefined,
                "id": "id-0",
                "modelId": "mock-model-id",
                "timestamp": 1970-01-01T00:00:00.000Z,
              },
              "type": "finish-step",
              "usage": {
                "inputTokenDetails": {
                  "cacheReadTokens": 0,
                  "cacheWriteTokens": 0,
                  "noCacheTokens": 10,
                },
                "inputTokens": 10,
                "outputTokenDetails": {
                  "reasoningTokens": 0,
                  "textTokens": 20,
                },
                "outputTokens": 20,
                "raw": undefined,
                "totalTokens": 30,
              },
            },
            {
              "finishReason": "content-filter",
              "rawFinishReason": "content-filter",
              "totalUsage": {
                "inputTokenDetails": {
                  "cacheReadTokens": 0,
                  "cacheWriteTokens": 0,
                  "noCacheTokens": 10,
                },
                "inputTokens": 10,
                "outputTokenDetails": {
                  "reasoningTokens": 0,
                  "textTokens": 20,
                },
                "outputTokens": 20,
                "totalTokens": 30,
              },
              "type": "finish",
            },
          ]
        `);
      });

      it('should ignore plain language models for result-based attempts', async () => {
        // Arrange
        const baseModel = MockLanguageModel.from({
          doStream: contentFilterStreamChunks,
        });
        const fallbackModel1 = MockLanguageModel.from({
          doStream: mockStreamChunks,
        });
        const fallbackModel2 = MockLanguageModel.from({
          doStream: mockStreamChunks,
        });

        const fallbackRetryable: ModelRetryable<LanguageModel> = (context) => {
          if (
            isResultAttempt(context.current) &&
            context.current.result.finishReason.unified === 'content-filter'
          ) {
            return { model: fallbackModel2, maxAttempts: 1 };
          }
          return undefined;
        };

        // Act
        const result = streamText({
          model: createRetryableModel({
            model: baseModel,
            retries: [
              fallbackModel1, // Language model should be skipped
              fallbackRetryable,
            ],
          }),
          prompt,
        });
        const chunks = await Iterables.toArray(result.fullStream);

        // Assert
        expect(baseModel.doStream).toHaveBeenCalledTimes(1);
        expect(fallbackModel1.doStream).toHaveBeenCalledTimes(0);
        expect(fallbackModel2.doStream).toHaveBeenCalledTimes(1);
        expect(chunks).toMatchInlineSnapshot(`
          [
            {
              "type": "start",
            },
            {
              "request": {
                "body": undefined,
                "messages": undefined,
              },
              "type": "start-step",
              "warnings": [],
            },
            {
              "id": "1",
              "type": "text-start",
            },
            {
              "id": "1",
              "providerMetadata": undefined,
              "text": "Hello",
              "type": "text-delta",
            },
            {
              "id": "1",
              "providerMetadata": undefined,
              "text": ", ",
              "type": "text-delta",
            },
            {
              "id": "1",
              "providerMetadata": undefined,
              "text": "world!",
              "type": "text-delta",
            },
            {
              "id": "1",
              "type": "text-end",
            },
            {
              "finishReason": "stop",
              "providerMetadata": {
                "testProvider": {
                  "testKey": "testValue",
                },
              },
              "rawFinishReason": "stop",
              "response": {
                "headers": undefined,
                "id": "id-0",
                "modelId": "mock-model-id",
                "timestamp": 1970-01-01T00:00:00.000Z,
              },
              "type": "finish-step",
              "usage": {
                "inputTokenDetails": {
                  "cacheReadTokens": 0,
                  "cacheWriteTokens": 0,
                  "noCacheTokens": 10,
                },
                "inputTokens": 10,
                "outputTokenDetails": {
                  "reasoningTokens": 0,
                  "textTokens": 20,
                },
                "outputTokens": 20,
                "raw": undefined,
                "totalTokens": 30,
              },
            },
            {
              "finishReason": "stop",
              "rawFinishReason": "stop",
              "totalUsage": {
                "inputTokenDetails": {
                  "cacheReadTokens": 0,
                  "cacheWriteTokens": 0,
                  "noCacheTokens": 10,
                },
                "inputTokens": 10,
                "outputTokenDetails": {
                  "reasoningTokens": 0,
                  "textTokens": 20,
                },
                "outputTokens": 20,
                "totalTokens": 30,
              },
              "type": "finish",
            },
          ]
        `);
      });

      it('should ignore static retries for result-based attempts', async () => {
        // Arrange
        const baseModel = MockLanguageModel.from({
          doStream: contentFilterStreamChunks,
        });
        const fallbackModel1 = MockLanguageModel.from({
          doStream: mockStreamChunks,
        });
        const fallbackModel2 = MockLanguageModel.from({
          doStream: mockStreamChunks,
        });

        const fallbackRetryable: ModelRetryable<LanguageModel> = (context) => {
          if (
            isResultAttempt(context.current) &&
            context.current.result.finishReason.unified === 'content-filter'
          ) {
            return { model: fallbackModel2, maxAttempts: 1 };
          }
          return undefined;
        };

        // Act
        const result = streamText({
          model: createRetryableModel({
            model: baseModel,
            retries: [
              { model: fallbackModel1 }, // Static retry should be skipped
              fallbackRetryable,
            ],
          }),
          prompt,
        });
        const chunks = await Iterables.toArray(result.fullStream);

        // Assert
        expect(baseModel.doStream).toHaveBeenCalledTimes(1);
        expect(fallbackModel1.doStream).toHaveBeenCalledTimes(0);
        expect(fallbackModel2.doStream).toHaveBeenCalledTimes(1);
        expect(chunks).toMatchInlineSnapshot(`
          [
            {
              "type": "start",
            },
            {
              "request": {
                "body": undefined,
                "messages": undefined,
              },
              "type": "start-step",
              "warnings": [],
            },
            {
              "id": "1",
              "type": "text-start",
            },
            {
              "id": "1",
              "providerMetadata": undefined,
              "text": "Hello",
              "type": "text-delta",
            },
            {
              "id": "1",
              "providerMetadata": undefined,
              "text": ", ",
              "type": "text-delta",
            },
            {
              "id": "1",
              "providerMetadata": undefined,
              "text": "world!",
              "type": "text-delta",
            },
            {
              "id": "1",
              "type": "text-end",
            },
            {
              "finishReason": "stop",
              "providerMetadata": {
                "testProvider": {
                  "testKey": "testValue",
                },
              },
              "rawFinishReason": "stop",
              "response": {
                "headers": undefined,
                "id": "id-0",
                "modelId": "mock-model-id",
                "timestamp": 1970-01-01T00:00:00.000Z,
              },
              "type": "finish-step",
              "usage": {
                "inputTokenDetails": {
                  "cacheReadTokens": 0,
                  "cacheWriteTokens": 0,
                  "noCacheTokens": 10,
                },
                "inputTokens": 10,
                "outputTokenDetails": {
                  "reasoningTokens": 0,
                  "textTokens": 20,
                },
                "outputTokens": 20,
                "raw": undefined,
                "totalTokens": 30,
              },
            },
            {
              "finishReason": "stop",
              "rawFinishReason": "stop",
              "totalUsage": {
                "inputTokenDetails": {
                  "cacheReadTokens": 0,
                  "cacheWriteTokens": 0,
                  "noCacheTokens": 10,
                },
                "inputTokens": 10,
                "outputTokenDetails": {
                  "reasoningTokens": 0,
                  "textTokens": 20,
                },
                "outputTokens": 20,
                "totalTokens": 30,
              },
              "type": "finish",
            },
          ]
        `);
      });
    });
  });

  describe('disabled', () => {
    it('should not retry when disabled is true', async () => {
      // Arrange
      const baseModel = MockLanguageModel.from({ doStream: retryableError });
      const fallbackModel = MockLanguageModel.from({
        doStream: mockStreamChunks,
      });

      const fallbackRetryable: ModelRetryable<LanguageModel> = () => {
        return { model: fallbackModel, maxAttempts: 1 };
      };

      const retryableModel = createRetryableModel({
        model: baseModel,
        retries: [fallbackRetryable],
        disabled: true,
      });

      // Act
      const result = streamText({
        model: retryableModel,
        prompt: 'Hello!',
        maxRetries: 0, // Disable AI SDK's own retry mechanism
      });

      const chunks = await Iterables.toArray(result.fullStream);

      // Assert
      expect(baseModel.doStream).toHaveBeenCalledTimes(1);
      expect(fallbackModel.doStream).toHaveBeenCalledTimes(0);

      // Check that an error chunk was emitted
      const errorChunk: any = chunks.find(
        (chunk: any) => chunk.type === 'error',
      );
      expect(errorChunk).toBeDefined();
      expect(errorChunk.error.message).toBe('Rate limit exceeded');
    });

    it('should retry when disabled is false', async () => {
      // Arrange
      const baseModel = MockLanguageModel.from({ doStream: retryableError });
      const fallbackModel = MockLanguageModel.from({
        doStream: mockStreamChunks,
      });

      const fallbackRetryable: ModelRetryable<LanguageModel> = () => {
        return { model: fallbackModel, maxAttempts: 1 };
      };

      const retryableModel = createRetryableModel({
        model: baseModel,
        retries: [fallbackRetryable],
        disabled: false,
      });

      // Act
      const result = streamText({
        model: retryableModel,
        prompt: 'Hello!',
      });

      const chunks = await Iterables.toArray(result.textStream);

      // Assert
      expect(baseModel.doStream).toHaveBeenCalledTimes(1);
      expect(fallbackModel.doStream).toHaveBeenCalledTimes(1);
      expect(chunks.join('')).toBe('Hello, world!');
    });
  });

  describe('onError', () => {
    it('should call onError handler when an error occurs', async () => {
      // Arrange
      const baseModel = MockLanguageModel.from({ doStream: retryableError });
      const fallbackModel = MockLanguageModel.from({
        doStream: mockStreamChunks,
      });
      const onErrorSpy = vi.fn<OnError>();

      // Act
      const retryableModel = createRetryableModel({
        model: baseModel,
        retries: [fallbackModel],
        onError: onErrorSpy,
      });

      const result = streamText({
        model: retryableModel,
        prompt,
      });

      await Iterables.toArray(result.fullStream);

      // Assert
      expect(onErrorSpy).toHaveBeenCalledTimes(1);

      const firstErrorCall = onErrorSpy.mock.calls[0]![0];
      expect(firstErrorCall.current.error).toBe(retryableError);
      expect(firstErrorCall.current.model).toBe(baseModel);
      expect(firstErrorCall.attempts.length).toBe(1);
      // expect(firstErrorCall.totalAttempts).toBe(1);
    });

    it('should call onError handler for each error in multiple attempts', async () => {
      // Arrange
      const baseModel = MockLanguageModel.from({ doStream: retryableError });
      const fallbackModel = MockLanguageModel.from({
        doStream: nonRetryableError,
      });
      const finalModel = MockLanguageModel.from({
        doStream: mockStreamChunks,
      });
      const onErrorSpy = vi.fn<OnError>();

      // Act
      const retryableModel = createRetryableModel({
        model: baseModel,
        retries: [fallbackModel, finalModel],
        onError: onErrorSpy,
      });

      const result = streamText({
        model: retryableModel,
        prompt,
      });

      await Iterables.toArray(result.fullStream);

      // Assert
      expect(onErrorSpy).toHaveBeenCalledTimes(2);

      // Check that onError was called for each error
      const firstErrorCall = onErrorSpy.mock.calls[0]![0];
      const secondErrorCall = onErrorSpy.mock.calls[1]![0];

      expect(firstErrorCall.current.error).toBe(retryableError);
      expect(firstErrorCall.current.model).toBe(baseModel);
      expect(firstErrorCall.attempts.length).toBe(1);
      // expect(firstErrorCall.totalAttempts).toBe(1);

      expect(secondErrorCall.current.error).toBe(nonRetryableError);
      expect(secondErrorCall.current.model).toBe(fallbackModel);
      expect(secondErrorCall.attempts.length).toBe(2);
      // expect(secondErrorCall.totalAttempts).toBe(2);
    });

    it('should NOT call onError handler when streaming succeeds', async () => {
      // Arrange
      const baseModel = MockLanguageModel.from({
        doStream: mockStreamChunks,
      });
      const onErrorSpy = vi.fn<OnError>();

      // Act
      const retryableModel = createRetryableModel({
        model: baseModel,
        retries: [],
        onError: onErrorSpy,
      });

      const result = streamText({
        model: retryableModel,
        prompt,
      });

      await Iterables.toArray(result.fullStream);

      // Assert
      expect(onErrorSpy).not.toHaveBeenCalled();
    });
  });

  describe('onRetry', () => {
    it('should call onRetry handler for error-based retries', async () => {
      // Arrange
      const baseModel = MockLanguageModel.from({ doStream: retryableError });
      const fallbackModel = MockLanguageModel.from({
        doStream: mockStreamChunks,
      });
      const onRetrySpy = vi.fn<OnRetry>();

      // Act
      const retryableModel = createRetryableModel({
        model: baseModel,
        retries: [fallbackModel],
        onRetry: onRetrySpy,
      });

      const result = streamText({
        model: retryableModel,
        prompt,
      });

      await Iterables.toArray(result.fullStream);

      // Assert
      expect(onRetrySpy).toHaveBeenCalledTimes(1);

      const firstRetryCall = onRetrySpy.mock.calls[0]![0];
      expect(isErrorAttempt(firstRetryCall.current)).toBe(true);
      if (isErrorAttempt(firstRetryCall.current)) {
        expect(firstRetryCall.current.error).toBe(retryableError);
      }
      expect(firstRetryCall.current.model).toBe(fallbackModel);
      expect(firstRetryCall.attempts.length).toBe(1);
      // expect(firstRetryCall.totalAttempts).toBe(1);
    });

    it('should call onRetry handler for each retry attempt', async () => {
      // Arrange
      const baseModel = MockLanguageModel.from({ doStream: retryableError });
      const fallbackModel1 = MockLanguageModel.from({
        doStream: nonRetryableError,
      });
      const fallbackModel2 = MockLanguageModel.from({
        doStream: mockStreamChunks,
      });
      const onRetrySpy = vi.fn<OnRetry>();

      // Act
      const retryableModel = createRetryableModel({
        model: baseModel,
        retries: [fallbackModel1, fallbackModel2],
        onRetry: onRetrySpy,
      });

      const result = streamText({
        model: retryableModel,
        prompt,
      });

      await Iterables.toArray(result.fullStream);

      // Assert
      expect(onRetrySpy).toHaveBeenCalledTimes(2);

      // Check that onRetry was called for each retry
      const firstRetryCall = onRetrySpy.mock.calls[0]![0];
      const secondRetryCall = onRetrySpy.mock.calls[1]![0];

      expect(isErrorAttempt(firstRetryCall.current)).toBe(true);
      if (isErrorAttempt(firstRetryCall.current)) {
        expect(firstRetryCall.current.error).toBe(retryableError);
      }
      expect(firstRetryCall.current.model).toBe(fallbackModel1);
      expect(firstRetryCall.attempts.length).toBe(1);
      // expect(firstRetryCall.totalAttempts).toBe(1);

      expect(isErrorAttempt(secondRetryCall.current)).toBe(true);
      if (isErrorAttempt(secondRetryCall.current)) {
        expect(secondRetryCall.current.error).toBe(nonRetryableError);
      }
      expect(secondRetryCall.current.model).toBe(fallbackModel2);
      expect(secondRetryCall.attempts.length).toBe(2);
      // expect(secondRetryCall.totalAttempts).toBe(2);
    });

    it('should NOT call onRetry on first attempt', async () => {
      // Arrange
      const baseModel = MockLanguageModel.from({
        doStream: mockStreamChunks,
      });
      const onRetrySpy = vi.fn<OnRetry>();

      // Act
      const retryableModel = createRetryableModel({
        model: baseModel,
        retries: [],
        onRetry: onRetrySpy,
      });

      const result = streamText({
        model: retryableModel,
        prompt,
      });

      await Iterables.toArray(result.fullStream);

      // Assert
      expect(onRetrySpy).not.toHaveBeenCalled();
    });
  });

  describe('onSuccess', () => {
    it('should call onSuccess with base model when no retry occurs', async () => {
      // Arrange
      const baseModel = MockLanguageModel.from({
        doStream: mockStreamChunks,
      });
      const onSuccessSpy = vi.fn<OnSuccess>();

      // Act
      const retryableModel = createRetryableModel({
        model: baseModel,
        retries: [],
        onSuccess: onSuccessSpy,
      });

      const result = streamText({
        model: retryableModel,
        prompt,
      });

      await Iterables.toArray(result.fullStream);

      // Assert
      expect(onSuccessSpy).toHaveBeenCalledTimes(1);

      const successCall = onSuccessSpy.mock.calls[0]![0];
      expect(successCall.current.type).toBe('success');
      expect(successCall.current.model).toBe(baseModel);
      expect(successCall.attempts.length).toBe(0);
    });

    it('should call onSuccess with fallback model after retry', async () => {
      // Arrange
      const baseModel = MockLanguageModel.from({ doStream: retryableError });
      const fallbackModel = MockLanguageModel.from({
        doStream: mockStreamChunks,
      });
      const onSuccessSpy = vi.fn<OnSuccess>();

      // Act
      const retryableModel = createRetryableModel({
        model: baseModel,
        retries: [fallbackModel],
        onSuccess: onSuccessSpy,
      });

      const result = streamText({
        model: retryableModel,
        prompt,
      });

      await Iterables.toArray(result.fullStream);

      // Assert
      expect(onSuccessSpy).toHaveBeenCalledTimes(1);

      const successCall = onSuccessSpy.mock.calls[0]![0];
      expect(successCall.current.type).toBe('success');
      expect(successCall.current.model).toBe(fallbackModel);
      expect(successCall.attempts.length).toBe(1);
    });

    it('should call onSuccess with no attempts when a retryable finish reason matches no retry', async () => {
      // Arrange
      const baseModel = MockLanguageModel.from({
        doStream: contentFilterStreamChunks,
      });
      const onSuccessSpy = vi.fn<OnSuccess>();

      // Act
      const result = streamText({
        model: createRetryableModel({
          model: baseModel,
          retries: [],
          onSuccess: onSuccessSpy,
        }),
        prompt,
      });

      await Iterables.toArray(result.fullStream);

      // Assert
      expect(onSuccessSpy).toHaveBeenCalledTimes(1);

      const successCall = onSuccessSpy.mock.calls[0]![0];
      expect(successCall.current.model).toBe(baseModel);
      expect(successCall.attempts.length).toBe(0);
    });
  });

  describe('onFailure', () => {
    it('should call onFailure when the initial stream fails with no retry', async () => {
      // Arrange
      const baseModel = MockLanguageModel.from({ doStream: nonRetryableError });
      const onFailureSpy = vi.fn<OnFailure>();

      // Act
      const result = streamText({
        model: createRetryableModel({
          model: baseModel,
          retries: [],
          onFailure: onFailureSpy,
        }),
        prompt,
      });
      await Iterables.toArray(result.fullStream);

      // Assert
      expect(onFailureSpy).toHaveBeenCalledTimes(1);

      const failureCall = onFailureSpy.mock.calls[0]![0];
      expect(failureCall.current.type).toBe('error');
      expect(failureCall.current.model).toBe(baseModel);
      expect(failureCall.attempts.length).toBe(1);
      expect(failureCall.error).toBe(nonRetryableError);
    });

    it('should call onFailure when a mid-stream error has no retry', async () => {
      // Arrange
      const baseModel = MockLanguageModel.from({
        doStream: errorStreamChunks(nonRetryableError),
      });
      const onFailureSpy = vi.fn<OnFailure>();

      // Act
      const result = streamText({
        model: createRetryableModel({
          model: baseModel,
          retries: [],
          onFailure: onFailureSpy,
        }),
        prompt,
      });
      const parts = await Iterables.toArray(result.fullStream);

      // Assert
      expect(errorFromChunks(parts)).toBe(nonRetryableError);
      expect(onFailureSpy).toHaveBeenCalledTimes(1);

      const failureCall = onFailureSpy.mock.calls[0]![0];
      expect(failureCall.current.type).toBe('error');
      expect(failureCall.current.model).toBe(baseModel);
      expect(failureCall.attempts.length).toBe(1);
      expect(failureCall.error).toBe(nonRetryableError);
    });

    it('should call onFailure when the re-stream after a retry fails', async () => {
      // Arrange — the base stream errors before content, so a retry is
      // selected, but creating the fallback's stream throws and nothing
      // matches it. That failure happens on the re-stream path, past the
      // terminal branches of the read loop.
      /**
       * Explicit model ids so the shared auto-increment counter is untouched
       * and the inline snapshots further down the file stay stable.
       */
      const baseModel = MockLanguageModel.from(
        { doStream: errorStreamChunks(retryableError) },
        { modelId: 're-stream-failure-base' },
      );
      const fallbackModel = MockLanguageModel.from(
        { doStream: nonRetryableError },
        { modelId: 're-stream-failure-fallback' },
      );
      const onFailureSpy = vi.fn<OnFailure>();

      // Act
      const result = streamText({
        model: createRetryableModel({
          model: baseModel,
          retries: [fallbackModel],
          onFailure: onFailureSpy,
        }),
        prompt,
      });
      const parts = await Iterables.toArray(result.fullStream);

      // Assert — surfaced as an error part, not a stream rejection, so the
      // consumer's own onError still fires.
      expect(RetryError.isInstance(errorFromChunks(parts))).toBe(true);
      expect(onFailureSpy).toHaveBeenCalledTimes(1);
      expect(onFailureSpy.mock.calls[0]![0].attempts.length).toBe(2);
    });

    it('should NOT call onFailure on a successful stream', async () => {
      // Arrange
      const baseModel = MockLanguageModel.from({
        doStream: mockStreamChunks,
      });
      const onFailureSpy = vi.fn<OnFailure>();
      const onSuccessSpy = vi.fn<OnSuccess>();

      // Act
      const result = streamText({
        model: createRetryableModel({
          model: baseModel,
          retries: [],
          onFailure: onFailureSpy,
          onSuccess: onSuccessSpy,
        }),
        prompt,
      });
      await Iterables.toArray(result.fullStream);

      // Assert
      expect(onSuccessSpy).toHaveBeenCalledTimes(1);
      expect(onFailureSpy).not.toHaveBeenCalled();
    });
  });

  describe('RetryableOptions', () => {
    describe('maxAttempts', () => {
      it('should try each model only once by default', async () => {
        // Arrange
        const baseModel = MockLanguageModel.from({ doStream: retryableError });
        const fallbackModel1 = MockLanguageModel.from({
          doStream: retryableError,
        });
        const fallbackModel2 = MockLanguageModel.from({
          doStream: retryableError,
        });
        const finalModel = MockLanguageModel.from({
          doStream: mockStreamChunks,
        });

        // Act
        const retryableModel = createRetryableModel({
          model: baseModel,
          retries: [
            fallbackModel1,
            { model: fallbackModel2 },
            async () => ({ model: finalModel }),
          ],
        });

        const result = streamText({
          model: retryableModel,
          prompt,
        });

        await Iterables.toArray(result.fullStream);

        // Assert
        expect(baseModel.doStream).toHaveBeenCalledTimes(1);
        expect(fallbackModel1.doStream).toHaveBeenCalledTimes(1);
        expect(fallbackModel2.doStream).toHaveBeenCalledTimes(1);
        expect(finalModel.doStream).toHaveBeenCalledTimes(1);
      });

      it('should try models multiple times if maxAttempts is set', async () => {
        // Arrange
        const baseModel = MockLanguageModel.from({ doStream: retryableError });
        const fallbackModel1 = MockLanguageModel.from({
          doStream: retryableError,
        });
        const fallbackModel2 = MockLanguageModel.from({
          doStream: retryableError,
        });
        const finalModel = MockLanguageModel.from({
          doStream: mockStreamChunks,
        });

        // Act
        const retryableModel = createRetryableModel({
          model: baseModel,
          retries: [
            // ModelRetryable<LanguageModel>  with different maxAttempts
            { model: fallbackModel1, maxAttempts: 2 },
            async () => ({ model: fallbackModel2, maxAttempts: 3 }),
            finalModel,
          ],
        });

        const result = streamText({
          model: retryableModel,
          prompt,
        });

        await Iterables.toArray(result.fullStream);

        // Assert
        expect(baseModel.doStream).toHaveBeenCalledTimes(1);
        expect(fallbackModel1.doStream).toHaveBeenCalledTimes(2);
        expect(fallbackModel2.doStream).toHaveBeenCalledTimes(3);
        expect(finalModel.doStream).toHaveBeenCalledTimes(1);
      });
    });

    describe('delay', () => {
      it('should apply delay before retrying', async () => {
        // Arrange
        vi.useFakeTimers();
        const baseModel = MockLanguageModel.from({ doStream: retryableError });
        const fallbackModel = MockLanguageModel.from({
          doStream: mockStreamChunks,
        });
        const delayMs = 100;

        // Act
        const result = streamText({
          model: createRetryableModel({
            model: baseModel,
            retries: [{ model: fallbackModel, delay: delayMs }],
          }),
          prompt,
        });

        const streamPromise = Iterables.toArray(result.fullStream);
        await vi.runAllTimersAsync();
        await streamPromise;

        // Assert
        expect(baseModel.doStream).toHaveBeenCalledTimes(1);
        expect(fallbackModel.doStream).toHaveBeenCalledTimes(1);

        vi.useRealTimers();
      });

      it('should apply different delays for multiple retries', async () => {
        // Arrange
        vi.useFakeTimers();
        const baseModel = MockLanguageModel.from({ doStream: retryableError });
        const fallbackModel1 = MockLanguageModel.from({
          doStream: retryableError,
        });
        const fallbackModel2 = MockLanguageModel.from({
          doStream: mockStreamChunks,
        });
        const delay1 = 50;
        const delay2 = 50;

        // Act
        const result = streamText({
          model: createRetryableModel({
            model: baseModel,
            retries: [
              { model: fallbackModel1, delay: delay1 },
              () => ({ model: fallbackModel2, delay: delay2 }),
            ],
          }),
          prompt,
        });

        const streamPromise = Iterables.toArray(result.fullStream);
        await vi.runAllTimersAsync();
        await streamPromise;

        // Assert
        expect(baseModel.doStream).toHaveBeenCalledTimes(1);
        expect(fallbackModel1.doStream).toHaveBeenCalledTimes(1);
        expect(fallbackModel2.doStream).toHaveBeenCalledTimes(1);

        vi.useRealTimers();
      });

      it('should not delay when delay is not specified', async () => {
        // Arrange
        const baseModel = MockLanguageModel.from({ doStream: retryableError });
        const fallbackModel = MockLanguageModel.from({
          doStream: mockStreamChunks,
        });

        // Act
        const result = streamText({
          model: createRetryableModel({
            model: baseModel,
            retries: [{ model: fallbackModel }],
          }),
          prompt,
        });

        await Iterables.toArray(result.fullStream);

        // Assert
        expect(baseModel.doStream).toHaveBeenCalledTimes(1);
        expect(fallbackModel.doStream).toHaveBeenCalledTimes(1);
      });
    });

    describe('providerOptions', () => {
      it('should override base model providerOptions with retry model providerOptions', async () => {
        // Arrange
        const baseModel = MockLanguageModel.from({ doStream: retryableError });
        const fallbackModel = MockLanguageModel.from({
          doStream: mockStreamChunks,
        });
        const originalProviderOptions = {
          openai: { user: 'original-user' },
        };
        const retryProviderOptions = { openai: { user: 'retry-user' } };

        // Act
        const result = streamText({
          model: createRetryableModel({
            model: baseModel,
            retries: [
              {
                model: fallbackModel,
                providerOptions: retryProviderOptions,
              },
            ],
          }),
          prompt,
          providerOptions: originalProviderOptions,
        });

        await Iterables.toArray(result.fullStream);

        // Assert
        expect(baseModel.doStream).toHaveBeenCalledTimes(1);
        expect(fallbackModel.doStream).toHaveBeenCalledTimes(1);
        expect(baseModel.doStream).toHaveBeenCalledWith(
          expect.objectContaining({
            providerOptions: originalProviderOptions,
          }),
        );
        expect(fallbackModel.doStream).toHaveBeenCalledWith(
          expect.objectContaining({
            providerOptions: retryProviderOptions,
          }),
        );
      });
    });

    describe('timeout', () => {
      it('should create fresh abort signal with specified timeout on retry', async () => {
        // Arrange
        let baseModelSignal: AbortSignal | undefined;
        let fallbackModelSignal: AbortSignal | undefined;
        // Use TimeoutError (from AbortSignal.timeout()) which should be retried,
        // as opposed to AbortError (from user cancellation) which should NOT be retried
        const timeoutError = Errors.timeout();

        const baseModel = MockLanguageModel.from({
          doStream: async (opts: LanguageModelCallOptions) => {
            baseModelSignal = opts.abortSignal;
            throw timeoutError;
          },
        });

        const fallbackModel = MockLanguageModel.from({
          doStream: async (opts: LanguageModelCallOptions) => {
            fallbackModelSignal = opts.abortSignal;
            // Verify the new signal is not aborted
            if (opts.abortSignal?.aborted) {
              throw new Error('Should not be aborted with fresh signal');
            }
            return {
              stream: Streams.from(mockStreamChunks),
            };
          },
        });

        // Create an already-aborted signal (simulates timeout that already fired)
        const controller = new AbortController();
        controller.abort(Errors.timeout());

        // Act
        const result = streamText({
          model: createRetryableModel({
            model: baseModel,
            retries: [
              {
                model: fallbackModel,
                timeout: 30000, // 30 second timeout for retry
              },
            ],
          }),
          prompt,
          abortSignal: controller.signal,
        });

        await Iterables.toArray(result.fullStream);

        // Assert
        expect(baseModel.doStream).toHaveBeenCalledTimes(1);
        expect(fallbackModel.doStream).toHaveBeenCalledTimes(1);

        // Base model should receive the original aborted signal
        expect(baseModelSignal?.aborted).toBe(true);

        // Fallback model should receive a fresh, non-aborted signal
        expect(fallbackModelSignal).toBeDefined();
        expect(fallbackModelSignal?.aborted).toBe(false);
        expect(baseModelSignal).not.toBe(fallbackModelSignal);
      });

      it('should not retry mid-stream abort when retry has no timeout', async () => {
        // Arrange — reproduces issue #39: stream creation succeeds but the
        // stream errors mid-flight while the inbound signal is aborted (as
        // happens when the AI SDK `timeout: { stepMs }` fires). Without a
        // fresh deadline on the retry, fallback can't recover, so we
        // rethrow rather than fire a misleading retry log.
        const abortError = Object.assign(
          new Error('The operation was aborted'),
          { name: 'AbortError' },
        );

        const baseModel = MockLanguageModel.from({
          doStream: [
            { type: 'stream-start', warnings: [] },
            { type: 'error', error: abortError },
          ],
        });
        const fallbackModel = MockLanguageModel.from({
          doStream: mockStreamChunks,
        });

        const onError = vi.fn();
        const onRetry = vi.fn();

        const controller = new AbortController();
        controller.abort();

        // Act
        const result = streamText({
          model: createRetryableModel({
            model: baseModel,
            retries: [fallbackModel],
            onError,
            onRetry,
          }),
          prompt,
          abortSignal: controller.signal,
          ...Options.stream,
        });

        await Iterables.toArray(result.fullStream);

        // Assert — fallback never runs and onRetry never fires; onError
        // still fires so operators see the underlying abort.
        expect(baseModel.doStream).toHaveBeenCalledTimes(1);
        expect(fallbackModel.doStream).toHaveBeenCalledTimes(0);
        expect(onError).toHaveBeenCalledTimes(1);
        expect(onRetry).toHaveBeenCalledTimes(0);
      });
    });

    describe('temperature', () => {
      it('should override temperature on retry after stream error', async () => {
        // Arrange
        // Base model returns a stream that errors before content starts
        const baseModel = MockLanguageModel.from({
          doStream: [
            { type: 'stream-start', warnings: [] },
            { type: 'error', error: new Error('Stream error') },
          ],
        });

        const fallbackModel = MockLanguageModel.from({
          doStream: mockStreamChunks,
        });

        const retryableModel = createRetryableModel({
          model: baseModel,
          retries: [{ model: fallbackModel, options: { temperature: 0.5 } }],
        });

        // Act
        const result = streamText({
          model: retryableModel,
          prompt: 'Hello!',
          temperature: 1.0,
        });

        await Iterables.toArray(result.fullStream);

        // Assert
        // Base model should be called with original temperature
        expect(baseModel.doStream).toHaveBeenCalledWith(
          expect.objectContaining({ temperature: 1.0 }),
        );

        // Fallback model should be called with overridden temperature
        expect(fallbackModel.doStream).toHaveBeenCalledWith(
          expect.objectContaining({ temperature: 0.5 }),
        );
      });
    });

    describe('topP', () => {
      it('should override topP on retry after stream error', async () => {
        // Arrange
        const baseModel = MockLanguageModel.from({
          doStream: [
            { type: 'stream-start', warnings: [] },
            { type: 'error', error: new Error('Stream error') },
          ],
        });

        const fallbackModel = MockLanguageModel.from({
          doStream: mockStreamChunks,
        });

        const retryableModel = createRetryableModel({
          model: baseModel,
          retries: [{ model: fallbackModel, options: { topP: 0.8 } }],
        });

        // Act
        const result = streamText({
          model: retryableModel,
          prompt: 'Hello!',
          topP: 1.0,
        });

        await Iterables.toArray(result.fullStream);

        // Assert
        expect(baseModel.doStream).toHaveBeenCalledWith(
          expect.objectContaining({ topP: 1.0 }),
        );
        expect(fallbackModel.doStream).toHaveBeenCalledWith(
          expect.objectContaining({ topP: 0.8 }),
        );
      });
    });

    describe('topK', () => {
      it('should override topK on retry after stream error', async () => {
        // Arrange
        const baseModel = MockLanguageModel.from({
          doStream: [
            { type: 'stream-start', warnings: [] },
            { type: 'error', error: new Error('Stream error') },
          ],
        });

        const fallbackModel = MockLanguageModel.from({
          doStream: mockStreamChunks,
        });

        const retryableModel = createRetryableModel({
          model: baseModel,
          retries: [{ model: fallbackModel, options: { topK: 10 } }],
        });

        // Act
        const result = streamText({
          model: retryableModel,
          prompt: 'Hello!',
          topK: 50,
        });

        await Iterables.toArray(result.fullStream);

        // Assert
        expect(baseModel.doStream).toHaveBeenCalledWith(
          expect.objectContaining({ topK: 50 }),
        );
        expect(fallbackModel.doStream).toHaveBeenCalledWith(
          expect.objectContaining({ topK: 10 }),
        );
      });
    });

    describe('maxOutputTokens', () => {
      it('should override maxOutputTokens on retry after stream error', async () => {
        // Arrange
        const baseModel = MockLanguageModel.from({
          doStream: [
            { type: 'stream-start', warnings: [] },
            { type: 'error', error: new Error('Stream error') },
          ],
        });

        const fallbackModel = MockLanguageModel.from({
          doStream: mockStreamChunks,
        });

        const retryableModel = createRetryableModel({
          model: baseModel,
          retries: [
            { model: fallbackModel, options: { maxOutputTokens: 500 } },
          ],
        });

        // Act
        const result = streamText({
          model: retryableModel,
          prompt: 'Hello!',
          maxOutputTokens: 1000,
        });

        await Iterables.toArray(result.fullStream);

        // Assert
        expect(baseModel.doStream).toHaveBeenCalledWith(
          expect.objectContaining({ maxOutputTokens: 1000 }),
        );
        expect(fallbackModel.doStream).toHaveBeenCalledWith(
          expect.objectContaining({ maxOutputTokens: 500 }),
        );
      });
    });

    describe('seed', () => {
      it('should override seed on retry after stream error', async () => {
        // Arrange
        const baseModel = MockLanguageModel.from({
          doStream: [
            { type: 'stream-start', warnings: [] },
            { type: 'error', error: new Error('Stream error') },
          ],
        });

        const fallbackModel = MockLanguageModel.from({
          doStream: mockStreamChunks,
        });

        const retryableModel = createRetryableModel({
          model: baseModel,
          retries: [{ model: fallbackModel, options: { seed: 42 } }],
        });

        // Act
        const result = streamText({
          model: retryableModel,
          prompt: 'Hello!',
          seed: 123,
        });

        await Iterables.toArray(result.fullStream);

        // Assert
        expect(baseModel.doStream).toHaveBeenCalledWith(
          expect.objectContaining({ seed: 123 }),
        );
        expect(fallbackModel.doStream).toHaveBeenCalledWith(
          expect.objectContaining({ seed: 42 }),
        );
      });
    });

    describe('stopSequences', () => {
      it('should override stopSequences on retry after stream error', async () => {
        // Arrange
        const baseModel = MockLanguageModel.from({
          doStream: [
            { type: 'stream-start', warnings: [] },
            { type: 'error', error: new Error('Stream error') },
          ],
        });

        const fallbackModel = MockLanguageModel.from({
          doStream: mockStreamChunks,
        });

        const retryableModel = createRetryableModel({
          model: baseModel,
          retries: [
            {
              model: fallbackModel,
              options: { stopSequences: ['RETRY_STOP'] },
            },
          ],
        });

        // Act
        const result = streamText({
          model: retryableModel,
          prompt: 'Hello!',
          stopSequences: ['ORIGINAL_STOP'],
        });

        await Iterables.toArray(result.fullStream);

        // Assert
        expect(baseModel.doStream).toHaveBeenCalledWith(
          expect.objectContaining({ stopSequences: ['ORIGINAL_STOP'] }),
        );
        expect(fallbackModel.doStream).toHaveBeenCalledWith(
          expect.objectContaining({ stopSequences: ['RETRY_STOP'] }),
        );
      });
    });

    describe('presencePenalty', () => {
      it('should override presencePenalty on retry after stream error', async () => {
        // Arrange
        const baseModel = MockLanguageModel.from({
          doStream: [
            { type: 'stream-start', warnings: [] },
            { type: 'error', error: new Error('Stream error') },
          ],
        });

        const fallbackModel = MockLanguageModel.from({
          doStream: mockStreamChunks,
        });

        const retryableModel = createRetryableModel({
          model: baseModel,
          retries: [
            { model: fallbackModel, options: { presencePenalty: 0.5 } },
          ],
        });

        // Act
        const result = streamText({
          model: retryableModel,
          prompt: 'Hello!',
          presencePenalty: 0.0,
        });

        await Iterables.toArray(result.fullStream);

        // Assert
        expect(baseModel.doStream).toHaveBeenCalledWith(
          expect.objectContaining({ presencePenalty: 0.0 }),
        );
        expect(fallbackModel.doStream).toHaveBeenCalledWith(
          expect.objectContaining({ presencePenalty: 0.5 }),
        );
      });
    });

    describe('frequencyPenalty', () => {
      it('should override frequencyPenalty on retry after stream error', async () => {
        // Arrange
        const baseModel = MockLanguageModel.from({
          doStream: [
            { type: 'stream-start', warnings: [] },
            { type: 'error', error: new Error('Stream error') },
          ],
        });

        const fallbackModel = MockLanguageModel.from({
          doStream: mockStreamChunks,
        });

        const retryableModel = createRetryableModel({
          model: baseModel,
          retries: [
            { model: fallbackModel, options: { frequencyPenalty: 0.8 } },
          ],
        });

        // Act
        const result = streamText({
          model: retryableModel,
          prompt: 'Hello!',
          frequencyPenalty: 0.2,
        });

        await Iterables.toArray(result.fullStream);

        // Assert
        expect(baseModel.doStream).toHaveBeenCalledWith(
          expect.objectContaining({ frequencyPenalty: 0.2 }),
        );
        expect(fallbackModel.doStream).toHaveBeenCalledWith(
          expect.objectContaining({ frequencyPenalty: 0.8 }),
        );
      });
    });

    describe('headers', () => {
      it('should override headers on retry after stream error', async () => {
        // Arrange
        const baseModel = MockLanguageModel.from({
          doStream: [
            { type: 'stream-start', warnings: [] },
            { type: 'error', error: new Error('Stream error') },
          ],
        });

        const fallbackModel = MockLanguageModel.from({
          doStream: mockStreamChunks,
        });

        const retryHeaders = { 'x-retry': 'retry' };

        const retryableModel = createRetryableModel({
          model: baseModel,
          retries: [
            { model: fallbackModel, options: { headers: retryHeaders } },
          ],
        });

        // Act
        const result = streamText({
          model: retryableModel,
          prompt: 'Hello!',
        });

        await Iterables.toArray(result.fullStream);

        // Assert
        expect(fallbackModel.doStream).toHaveBeenCalledWith(
          expect.objectContaining({
            headers: expect.objectContaining({
              'x-retry': 'retry',
            }),
          }),
        );
      });
    });
  });

  describe('RetryError', () => {
    it('should throw RetryError when all retry attempts are exhausted', async () => {
      // Arrange
      const baseModel = MockLanguageModel.from({ doStream: retryableError });
      const fallbackModel1 = MockLanguageModel.from({
        doStream: nonRetryableError,
      });
      const fallbackModel2 = MockLanguageModel.from({
        doStream: retryableError,
      });

      const retryableModel = createRetryableModel({
        model: baseModel,
        retries: [fallbackModel1, fallbackModel2],
      });

      const result = streamText({
        model: retryableModel,
        prompt,
      });

      // Act
      const chunks = await Iterables.toArray(result.fullStream);

      // Assert
      expect(baseModel.doStream).toHaveBeenCalledTimes(1);
      expect(fallbackModel1.doStream).toHaveBeenCalledTimes(1);
      expect(fallbackModel2.doStream).toHaveBeenCalledTimes(1);
      expect(chunks).toMatchInlineSnapshot(`
        [
          {
            "type": "start",
          },
          {
            "error": [AI_RetryError: Failed after 3 attempts. Last error: AI_APICallError: Rate limit exceeded],
            "type": "error",
          },
        ]
      `);

      const errorChunk = errorFromChunks(chunks);
      expect(errorChunk).toBeDefined();
      expect(errorChunk).toBeInstanceOf(RetryError);

      const retryError = errorChunk as RetryError;
      expect(retryError.reason).toBe('maxRetriesExceeded');
      expect(retryError.errors).toHaveLength(3);
      expect(retryError.errors[0]).toBe(retryableError);
      expect(retryError.errors[1]).toBe(nonRetryableError);
      expect(retryError.errors[2]).toBe(retryableError);
    });

    it('should throw original error directly on first attempt with no retryables', async () => {
      // Arrange
      const baseModel = MockLanguageModel.from({ doStream: retryableError });

      const retryableModel = createRetryableModel({
        model: baseModel,
        retries: [], // No retry models
      });

      const result = streamText({
        model: retryableModel,
        prompt,
        maxRetries: 0, // No automatic retries
      });

      // Act
      const chunks = await Iterables.toArray(result.fullStream);

      // Assert
      expect(baseModel.doStream).toHaveBeenCalledTimes(1);
      expect(errorFromChunks(chunks)).toBe(retryableError);
      expect(chunks).toMatchInlineSnapshot(`
        [
          {
            "type": "start",
          },
          {
            "error": [AI_APICallError: Rate limit exceeded],
            "type": "error",
          },
        ]
      `);
    });

    it('should throw original error directly when retryable returns undefined', async () => {
      // Arrange
      const baseModel = MockLanguageModel.from({ doStream: nonRetryableError });

      const retryableModel = createRetryableModel({
        model: baseModel,
        retries: [() => undefined],
      });

      const result = streamText({
        model: retryableModel,
        prompt,
        maxRetries: 0, // No automatic retries
      });

      // Act
      const chunks = await Iterables.toArray(result.fullStream);

      // Assert
      expect(baseModel.doStream).toHaveBeenCalledTimes(1);
      expect(errorFromChunks(chunks)).toBe(nonRetryableError);
      expect(chunks).toMatchInlineSnapshot(`
        [
          {
            "type": "start",
          },
          {
            "error": [AI_APICallError: Unauthorized],
            "type": "error",
          },
        ]
      `);
    });
  });

  describe(`reset`, () => {
    describe(`after-request (default)`, () => {
      it(`should reset to base model on every request`, async () => {
        // Arrange
        let callCount = 0;
        const baseModel = MockLanguageModel.from({
          doStream: async () => {
            callCount++;
            if (callCount <= 1) throw retryableError;
            return { stream: Streams.from(mockStreamChunks) };
          },
        });
        const fallbackModel = MockLanguageModel.from({
          doStream: async () => ({
            stream: Streams.from(mockStreamChunks),
          }),
        });

        const retryableModel = createRetryableModel({
          model: baseModel,
          retries: [{ model: fallbackModel, maxAttempts: 1 }],
        });

        // Act — first request: base fails, fallback succeeds
        const result1 = streamText({ model: retryableModel, prompt });
        const chunks1 = await Iterables.toArray(result1.fullStream);

        // Act — second request: base model is used again (reset), succeeds this time
        const result2 = streamText({ model: retryableModel, prompt });
        const chunks2 = await Iterables.toArray(result2.fullStream);

        // Assert
        expect(chunksToText(chunks1)).toBe(mockResultText);
        expect(chunksToText(chunks2)).toBe(mockResultText);
        expect(baseModel.doStream).toHaveBeenCalledTimes(2);
        expect(fallbackModel.doStream).toHaveBeenCalledTimes(1);
      });
    });

    describe(`after-N-requests`, () => {
      it(`should use sticky model for N subsequent requests then reset`, async () => {
        // Arrange
        const baseModel = MockLanguageModel.from({ doStream: retryableError });
        const fallbackModel = MockLanguageModel.from({
          doStream: async () => ({
            stream: Streams.from(mockStreamChunks),
          }),
        });

        const retryableModel = createRetryableModel({
          model: baseModel,
          retries: [{ model: fallbackModel, maxAttempts: 1 }],
          reset: `after-2-requests`,
        });

        // Act — request 1: base fails, fallback succeeds → sticky set
        const result1 = streamText({ model: retryableModel, prompt });
        const chunks1 = await Iterables.toArray(result1.fullStream);

        // Act — request 2: sticky (fallback) used directly
        const result2 = streamText({ model: retryableModel, prompt });
        const chunks2 = await Iterables.toArray(result2.fullStream);

        // Act — request 3: sticky (fallback) used directly (last sticky request)
        const result3 = streamText({ model: retryableModel, prompt });
        const chunks3 = await Iterables.toArray(result3.fullStream);

        // Act — request 4: sticky expired, back to base model → base fails, fallback retried
        const result4 = streamText({ model: retryableModel, prompt });
        const chunks4 = await Iterables.toArray(result4.fullStream);

        // Assert
        expect(chunksToText(chunks1)).toBe(mockResultText);
        expect(chunksToText(chunks2)).toBe(mockResultText);
        expect(chunksToText(chunks3)).toBe(mockResultText);
        expect(chunksToText(chunks4)).toBe(mockResultText);
        // Request 1: base(1) + fallback(1), Request 2: fallback(1), Request 3: fallback(1),
        // Request 4: base(1) + fallback(1)
        expect(baseModel.doStream).toHaveBeenCalledTimes(2);
        expect(fallbackModel.doStream).toHaveBeenCalledTimes(4);
      });

      it(`should not set sticky model when no retry occurred`, async () => {
        // Arrange
        const baseModel = MockLanguageModel.from({
          doStream: async () => ({
            stream: Streams.from(mockStreamChunks),
          }),
        });
        const fallbackModel = MockLanguageModel.from({
          doStream: async () => ({
            stream: Streams.from(mockStreamChunks),
          }),
        });

        const retryableModel = createRetryableModel({
          model: baseModel,
          retries: [{ model: fallbackModel, maxAttempts: 1 }],
          reset: `after-3-requests`,
        });

        // Act
        const r1 = streamText({ model: retryableModel, prompt });
        await Iterables.toArray(r1.fullStream);
        const r2 = streamText({ model: retryableModel, prompt });
        await Iterables.toArray(r2.fullStream);

        // Assert — base model used both times, no fallback
        expect(baseModel.doStream).toHaveBeenCalledTimes(2);
        expect(fallbackModel.doStream).toHaveBeenCalledTimes(0);
      });

      it(`should reset counter when a new sticky model is set`, async () => {
        // Arrange
        let fallback1CallCount = 0;
        const baseModel = MockLanguageModel.from({ doStream: retryableError });
        const fallbackModel1 = MockLanguageModel.from({
          doStream: async () => {
            fallback1CallCount++;
            /** Fail on second use (when used as sticky on request 2) */
            if (fallback1CallCount >= 2) throw retryableError;
            return { stream: Streams.from(mockStreamChunks) };
          },
        });
        const fallbackModel2 = MockLanguageModel.from({
          doStream: async () => ({
            stream: Streams.from(mockStreamChunks),
          }),
        });

        const retryableModel = createRetryableModel({
          model: baseModel,
          retries: [
            { model: fallbackModel1, maxAttempts: 1 },
            { model: fallbackModel2, maxAttempts: 1 },
          ],
          reset: `after-2-requests`,
        });

        // Act — request 1: base fails → fallback1 succeeds → sticky = fallback1
        const r1 = streamText({ model: retryableModel, prompt });
        await Iterables.toArray(r1.fullStream);

        // Act — request 2: sticky (fallback1) fails → fallback2 succeeds → sticky = fallback2, counter resets to 2
        const r2 = streamText({ model: retryableModel, prompt });
        await Iterables.toArray(r2.fullStream);

        // Act — request 3: sticky (fallback2) used directly (remaining = 1)
        const r3 = streamText({ model: retryableModel, prompt });
        await Iterables.toArray(r3.fullStream);

        // Act — request 4: sticky (fallback2) used directly (remaining = 0)
        const r4 = streamText({ model: retryableModel, prompt });
        await Iterables.toArray(r4.fullStream);

        // Assert
        expect(baseModel.doStream).toHaveBeenCalledTimes(1);
        expect(fallbackModel1.doStream).toHaveBeenCalledTimes(2);
        expect(fallbackModel2.doStream).toHaveBeenCalledTimes(3);
      });
    });

    describe(`after-N-seconds`, () => {
      it(`should use sticky model within time window then reset`, async () => {
        // Arrange
        vi.useFakeTimers();

        const baseModel = MockLanguageModel.from({ doStream: retryableError });
        const fallbackModel = MockLanguageModel.from({
          doStream: async () => ({
            stream: Streams.from(mockStreamChunks),
          }),
        });

        const retryableModel = createRetryableModel({
          model: baseModel,
          retries: [{ model: fallbackModel, maxAttempts: 1 }],
          reset: `after-5-seconds`,
        });

        // Act — request 1: base fails, fallback succeeds → sticky set
        const r1 = streamText({ model: retryableModel, prompt });
        await Iterables.toArray(r1.fullStream);

        // Act — request 2: within 5s, sticky (fallback) used directly
        vi.advanceTimersByTime(2_000);
        const r2 = streamText({ model: retryableModel, prompt });
        await Iterables.toArray(r2.fullStream);

        // Act — request 3: advance past 5s, sticky expired → back to base → fails → fallback
        vi.advanceTimersByTime(4_000);
        const r3 = streamText({ model: retryableModel, prompt });
        await Iterables.toArray(r3.fullStream);

        // Assert
        expect(baseModel.doStream).toHaveBeenCalledTimes(2);
        expect(fallbackModel.doStream).toHaveBeenCalledTimes(3);

        vi.useRealTimers();
      });
    });
  });
});

/**
 * Kept out of the blocks above on purpose: every `MockLanguageModel.from()`
 * consumes a shared model-id counter that snapshots elsewhere in this file
 * assert on, so a test added in the middle renumbers them.
 */
describe('streamText onSuccess and the commit boundary', () => {
  it('should not call onSuccess when the stream carries an error part', async () => {
    // Arrange — the error arrives after content, so the attempt is committed
    // and the error is forwarded to the consumer rather than retried. The
    // stream still closes normally, which is what made this look successful.
    const baseModel = MockLanguageModel.from({
      doStream: [
        Language.streamStart(),
        ...Language.streamText('Par', { id: '1' }),
        Language.streamError(retryableError),
      ],
    });
    const onSuccessSpy = vi.fn<OnSuccess>();

    // Act
    const result = streamText({
      model: createRetryableModel({
        model: baseModel,
        retries: [],
        onSuccess: onSuccessSpy,
      }),
      prompt,
      onError: () => {},
    });
    const chunks = await Streams.toArray(result.fullStream);

    // Assert — the consumer saw the failure, so this was not a success.
    expect(chunks.some((chunk) => chunk.type === 'error')).toBe(true);
    expect(onSuccessSpy).not.toHaveBeenCalled();
  });

  it('should still call onSuccess when the stream ends cleanly', async () => {
    // Arrange — the counterpart, so the guard cannot simply silence it.
    const baseModel = MockLanguageModel.from({ doStream: mockStreamChunks });
    const onSuccessSpy = vi.fn<OnSuccess>();

    // Act
    const result = streamText({
      model: createRetryableModel({
        model: baseModel,
        retries: [],
        onSuccess: onSuccessSpy,
      }),
      prompt,
    });
    await Streams.toArray(result.fullStream);

    // Assert
    expect(onSuccessSpy).toHaveBeenCalledTimes(1);
  });
});

describe('streamText abort passthrough', () => {
  /**
   * A model stream that emits `parts` and then fails the way a provider's
   * fetch body does when the call is aborted: the stream rejects with the
   * signal's reason, an `AbortError` for a cancel and a `TimeoutError` for a
   * deadline. `afterParts` runs once the parts are enqueued, so a test can
   * abort before any content reaches the consumer.
   */
  const abortableModel = (
    parts: Array<LanguageModelStreamPart>,
    afterParts?: () => void,
  ) =>
    MockLanguageModel.from({
      doStream: async (opts: LanguageModelCallOptions) => ({
        stream: new ReadableStream<LanguageModelStreamPart>({
          start(controller) {
            for (const part of parts) controller.enqueue(part);
            const onAbort = () => controller.error(opts.abortSignal?.reason);
            if (opts.abortSignal?.aborted) onAbort();
            else
              opts.abortSignal?.addEventListener('abort', onAbort, {
                once: true,
              });
            afterParts?.();
          },
        }),
      }),
    });

  const contentParts: Array<LanguageModelStreamPart> = [
    Language.streamStart(),
    { type: 'text-start', id: '0' },
    { type: 'text-delta', id: '0', delta: 'Hello' },
  ];

  /**
   * Read the full stream and abort the call once the first text delta
   * arrives, i.e. after the attempt has committed.
   */
  const consumeAndAbortOnContent = async (
    fullStream: AsyncIterable<{ type: string }>,
    controller: AbortController,
    reason?: unknown,
  ) => {
    const types: Array<string> = [];
    for await (const part of fullStream) {
      types.push(part.type);
      if (part.type === 'text-delta') controller.abort(reason);
    }
    return types;
  };

  /**
   * Read a model stream to its end, recording each part type in `types` and
   * aborting the call once the first text delta arrives. Rejects when the
   * stream does.
   */
  const readModelStream = async (
    stream: ReadableStream<LanguageModelStreamPart>,
    types: Array<string>,
    controller?: AbortController,
    reason?: unknown,
  ) => {
    for await (const part of stream) {
      types.push(part.type);
      if (part.type === 'text-delta') controller?.abort(reason);
    }
  };

  /** The reason a deadline aborts with, as `AbortSignal.timeout()` does. */
  const timeoutError = () =>
    new DOMException('The operation timed out.', 'TimeoutError');

  const callOptions = (
    abortSignal?: AbortSignal,
  ): LanguageModelCallOptions => ({
    prompt: [{ role: 'user', content: [{ type: 'text', text: prompt }] }],
    abortSignal,
  });

  /**
   * A model that streams content and, once the call is aborted, sends an
   * abort error part. `afterAbortPart` runs after that part is enqueued, so a
   * test can make the model keep going. Like a real provider, the model stops
   * once its stream is cancelled, which `isCancelled` reports.
   */
  const abortPartModel = (
    afterAbortPart: (
      controller: ReadableStreamDefaultController,
      isCancelled: () => boolean,
    ) => void,
  ) =>
    MockLanguageModel.from({
      doStream: async (opts: LanguageModelCallOptions) => {
        let cancelled = false;
        return {
          stream: new ReadableStream<LanguageModelStreamPart>({
            cancel() {
              cancelled = true;
            },
            start(controller) {
              for (const part of contentParts) controller.enqueue(part);
              opts.abortSignal?.addEventListener(
                'abort',
                () => {
                  controller.enqueue({
                    type: 'error',
                    error: new DOMException(
                      'The operation was aborted.',
                      'AbortError',
                    ),
                  });
                  afterAbortPart(controller, () => cancelled);
                },
                { once: true },
              );
            },
          }),
        };
      },
    });

  describe('wrapped model stream', () => {
    it('should reject with the abort error when the call aborts after content has streamed', async () => {
      // Arrange
      const controller = new AbortController();
      const model = createRetryableModel({
        model: abortableModel(contentParts),
        retries: [MockLanguageModel.from({ doStream: mockStreamChunks })],
      });
      const types: Array<string> = [];

      // Act
      const { stream } = await model.doStream(callOptions(controller.signal));
      const result = readModelStream(stream, types, controller);

      // Assert
      await expect(result).rejects.toMatchObject({ name: 'AbortError' });
      expect(types.includes('error')).toBe(false);
    });

    it('should reject with the abort error when the call aborts before content and no retry matches', async () => {
      // Arrange
      const controller = new AbortController();
      const model = createRetryableModel({
        model: abortableModel([Language.streamStart()], () =>
          controller.abort(),
        ),
        retries: [],
      });
      const types: Array<string> = [];

      // Act
      const { stream } = await model.doStream(callOptions(controller.signal));
      const result = readModelStream(stream, types);

      // Assert
      await expect(result).rejects.toMatchObject({ name: 'AbortError' });
      expect(types.includes('error')).toBe(false);
    });

    it('should reject with the abort error, not a RetryError, when the call aborts after a failover', async () => {
      // Arrange
      const controller = new AbortController();
      const baseModel = MockLanguageModel.from({
        doStream: async () => {
          throw retryableError;
        },
      });
      const fallbackModel = abortableModel([Language.streamStart()], () =>
        controller.abort(),
      );
      const onFailure = vi.fn<OnFailure>();
      const model = createRetryableModel({
        model: baseModel,
        retries: [fallbackModel],
        onFailure,
      });
      const types: Array<string> = [];

      // Act
      const { stream } = await model.doStream(callOptions(controller.signal));
      const result = readModelStream(stream, types);

      // Assert
      await expect(result).rejects.toMatchObject({ name: 'AbortError' });
      expect(types.includes('error')).toBe(false);
      expect(fallbackModel.doStream).toHaveBeenCalledTimes(1);
      const [failure] = onFailure.mock.calls[0]!;
      expect(failure.attempts.length).toBe(2);
      expect(failure.error).toMatchObject({ name: 'AbortError' });
    });

    it('should reject with the abort error when the call aborts before content and the retry would die on the aborted signal', async () => {
      // Arrange
      const controller = new AbortController();
      const fallbackModel = MockLanguageModel.from({
        doStream: mockStreamChunks,
      });
      const model = createRetryableModel({
        model: abortableModel([Language.streamStart()], () =>
          controller.abort(),
        ),
        retries: [fallbackModel],
      });
      const types: Array<string> = [];

      // Act
      const { stream } = await model.doStream(callOptions(controller.signal));
      const result = readModelStream(stream, types);

      // Assert
      await expect(result).rejects.toMatchObject({ name: 'AbortError' });
      expect(types.includes('error')).toBe(false);
      expect(fallbackModel.doStream).toHaveBeenCalledTimes(0);
    });

    it('should reject with the abort error when the model forwards an abort error part after content', async () => {
      // Arrange
      const controller = new AbortController();
      const model = createRetryableModel({
        model: abortPartModel((streamController) => streamController.close()),
        retries: [],
      });
      const types: Array<string> = [];

      // Act
      const { stream } = await model.doStream(callOptions(controller.signal));
      const result = readModelStream(stream, types, controller);

      // Assert
      await expect(result).rejects.toMatchObject({ name: 'AbortError' });
      expect(types.includes('error')).toBe(false);
    });

    it('should reject with the abort error part at once, even when the model keeps streaming and then fails', async () => {
      // Arrange
      const controller = new AbortController();
      const model = createRetryableModel({
        model: abortPartModel((streamController, isCancelled) => {
          setTimeout(() => {
            if (isCancelled()) return;
            streamController.enqueue({
              type: 'text-delta',
              id: '0',
              delta: 'late',
            });
            streamController.error(new TypeError('terminated'));
          }, 0);
        }),
        retries: [],
      });
      const types: Array<string> = [];

      // Act
      const { stream } = await model.doStream(callOptions(controller.signal));
      const result = readModelStream(stream, types, controller);

      // Assert
      await expect(result).rejects.toMatchObject({ name: 'AbortError' });
      expect(types.filter((type) => type === 'text-delta').length).toBe(1);
      expect(types.includes('error')).toBe(false);
    });

    it('should reject with the timeout error when the call deadline fires after content has streamed', async () => {
      // Arrange
      const controller = new AbortController();
      const model = createRetryableModel({
        model: abortableModel(contentParts),
        retries: [],
      });
      const types: Array<string> = [];

      // Act
      const { stream } = await model.doStream(callOptions(controller.signal));
      const result = readModelStream(stream, types, controller, timeoutError());

      // Assert
      await expect(result).rejects.toMatchObject({ name: 'TimeoutError' });
      expect(types.includes('error')).toBe(false);
    });

    it('should reject with the timeout error when the call deadline fires before content and no retry matches', async () => {
      // Arrange
      const controller = new AbortController();
      const model = createRetryableModel({
        model: abortableModel([Language.streamStart()], () =>
          controller.abort(timeoutError()),
        ),
        retries: [],
      });
      const types: Array<string> = [];

      // Act
      const { stream } = await model.doStream(callOptions(controller.signal));
      const result = readModelStream(stream, types);

      // Assert
      await expect(result).rejects.toMatchObject({ name: 'TimeoutError' });
      expect(types.includes('error')).toBe(false);
    });

    it('should forward a non-abort error after content as an error part', async () => {
      // Arrange
      const baseModel = MockLanguageModel.from({
        doStream: async () => ({
          stream: new ReadableStream<LanguageModelStreamPart>({
            start(controller) {
              for (const part of contentParts) controller.enqueue(part);
              controller.error(new Error('connection reset'));
            },
          }),
        }),
      });
      const model = createRetryableModel({ model: baseModel, retries: [] });
      const types: Array<string> = [];

      // Act
      const { stream } = await model.doStream(callOptions());
      const result = readModelStream(stream, types);

      // Assert
      await expect(result).resolves.toBe(undefined);
      expect(types.at(-1)).toBe('error');
    });
  });

  it('should call only onAbort for the unwrapped model (baseline)', async () => {
    // Arrange
    const controller = new AbortController();
    const onError = vi.fn();
    const onAbort = vi.fn();

    // Act
    const result = streamText({
      model: abortableModel(contentParts),
      prompt,
      abortSignal: controller.signal,
      onError,
      onAbort,
    });
    const types = await consumeAndAbortOnContent(result.fullStream, controller);

    // Assert
    expect(onAbort).toHaveBeenCalledTimes(1);
    expect(onError).toHaveBeenCalledTimes(0);
    expect(types.includes('error')).toBe(false);
  });

  it('should call only onAbort when the call aborts after content has streamed', async () => {
    // Arrange
    const controller = new AbortController();
    const onError = vi.fn();
    const onAbort = vi.fn();
    const onFailure = vi.fn<OnFailure>();
    const fallbackModel = MockLanguageModel.from({
      doStream: mockStreamChunks,
    });

    // Act
    const result = streamText({
      model: createRetryableModel({
        model: abortableModel(contentParts),
        retries: [fallbackModel],
        onFailure,
      }),
      prompt,
      abortSignal: controller.signal,
      onError,
      onAbort,
    });
    const types = await consumeAndAbortOnContent(result.fullStream, controller);

    // Assert
    expect(onAbort).toHaveBeenCalledTimes(1);
    expect(onError).toHaveBeenCalledTimes(0);
    expect(types.includes('error')).toBe(false);
    expect(fallbackModel.doStream).toHaveBeenCalledTimes(0);
  });

  it('should call only onAbort when the call aborts before content and no retry matches', async () => {
    // Arrange
    const controller = new AbortController();
    const onError = vi.fn();
    const onAbort = vi.fn();
    const onFailure = vi.fn<OnFailure>();

    // Act
    const result = streamText({
      model: createRetryableModel({
        model: abortableModel([Language.streamStart()], () =>
          controller.abort(),
        ),
        retries: [],
        onFailure,
      }),
      prompt,
      abortSignal: controller.signal,
      onError,
      onAbort,
    });
    const types = await Iterables.toArray(result.fullStream);

    // Assert
    expect(onAbort).toHaveBeenCalledTimes(1);
    expect(onError).toHaveBeenCalledTimes(0);
    expect(types.some((part) => part.type === 'error')).toBe(false);
    expect(onFailure).toHaveBeenCalledTimes(1);
  });

  it('should call only onAbort when the call aborts before content and the retry would die on the aborted signal', async () => {
    // Arrange
    const controller = new AbortController();
    const onError = vi.fn();
    const onAbort = vi.fn();
    const fallbackModel = MockLanguageModel.from({
      doStream: mockStreamChunks,
    });

    // Act
    const result = streamText({
      model: createRetryableModel({
        model: abortableModel([Language.streamStart()], () =>
          controller.abort(),
        ),
        retries: [fallbackModel],
      }),
      prompt,
      abortSignal: controller.signal,
      onError,
      onAbort,
    });
    const types = await Iterables.toArray(result.fullStream);

    // Assert
    expect(onAbort).toHaveBeenCalledTimes(1);
    expect(onError).toHaveBeenCalledTimes(0);
    expect(types.some((part) => part.type === 'error')).toBe(false);
    expect(fallbackModel.doStream).toHaveBeenCalledTimes(0);
  });

  it('should call only onAbort when the model forwards an abort error part after content', async () => {
    // Arrange
    const controller = new AbortController();
    const onError = vi.fn();
    const onAbort = vi.fn();
    const abortError = new DOMException(
      'The operation was aborted.',
      'AbortError',
    );
    const baseModel = MockLanguageModel.from({
      doStream: async (opts: LanguageModelCallOptions) => ({
        stream: new ReadableStream<LanguageModelStreamPart>({
          start(streamController) {
            for (const part of contentParts) streamController.enqueue(part);
            opts.abortSignal?.addEventListener(
              'abort',
              () => {
                streamController.enqueue({ type: 'error', error: abortError });
                streamController.close();
              },
              { once: true },
            );
          },
        }),
      }),
    });

    // Act
    const result = streamText({
      model: createRetryableModel({ model: baseModel, retries: [] }),
      prompt,
      abortSignal: controller.signal,
      onError,
      onAbort,
    });
    const types = await consumeAndAbortOnContent(result.fullStream, controller);

    // Assert
    expect(onAbort).toHaveBeenCalledTimes(1);
    expect(onError).toHaveBeenCalledTimes(0);
    expect(types.includes('error')).toBe(false);
  });

  it('should still call onError for a non-abort error after content', async () => {
    // Arrange
    const onError = vi.fn();
    const onAbort = vi.fn();
    const baseModel = MockLanguageModel.from({
      doStream: async () => ({
        stream: new ReadableStream<LanguageModelStreamPart>({
          start(controller) {
            for (const part of contentParts) controller.enqueue(part);
            controller.error(new Error('connection reset'));
          },
        }),
      }),
    });

    // Act
    const result = streamText({
      model: createRetryableModel({ model: baseModel, retries: [] }),
      prompt,
      onError,
      onAbort,
    });
    const types = await Iterables.toArray(result.fullStream);

    // Assert
    expect(onError).toHaveBeenCalledTimes(1);
    expect(onAbort).toHaveBeenCalledTimes(0);
    expect(types.some((part) => part.type === 'error')).toBe(true);
  });

  it('should call only onAbort when the call deadline fires after content has streamed', async () => {
    // Arrange
    const controller = new AbortController();
    const onError = vi.fn();
    const onAbort = vi.fn();

    // Act
    const result = streamText({
      model: createRetryableModel({
        model: abortableModel(contentParts),
        retries: [],
      }),
      prompt,
      abortSignal: controller.signal,
      onError,
      onAbort,
    });
    const types = await consumeAndAbortOnContent(
      result.fullStream,
      controller,
      timeoutError(),
    );

    // Assert
    expect(onAbort).toHaveBeenCalledTimes(1);
    expect(onError).toHaveBeenCalledTimes(0);
    expect(types.includes('error')).toBe(false);
  });

  it('should call only onAbort when a streamText timeout fires after content has streamed', async () => {
    // Arrange: the model streams content and then stalls until the deadline
    // aborts it.
    const onError = vi.fn();
    const onAbort = vi.fn();

    // Act
    const result = streamText({
      model: createRetryableModel({
        model: abortableModel(contentParts),
        retries: [],
      }),
      prompt,
      timeout: { totalMs: 50 },
      onError,
      onAbort,
    });
    const types = await Iterables.toArray(result.fullStream);

    // Assert
    expect(onAbort).toHaveBeenCalledTimes(1);
    expect(onError).toHaveBeenCalledTimes(0);
    expect(types.some((part) => part.type === 'error')).toBe(false);
  });

  it('should still call onError for an abort error while the call signal is not aborted', async () => {
    // Arrange
    const onError = vi.fn();
    const onAbort = vi.fn();
    const baseModel = MockLanguageModel.from({
      doStream: async () => ({
        stream: new ReadableStream<LanguageModelStreamPart>({
          start(controller) {
            for (const part of contentParts) controller.enqueue(part);
            controller.error(
              new DOMException('The operation timed out.', 'TimeoutError'),
            );
          },
        }),
      }),
    });

    // Act
    const result = streamText({
      model: createRetryableModel({ model: baseModel, retries: [] }),
      prompt,
      onError,
      onAbort,
    });
    const types = await Iterables.toArray(result.fullStream);

    // Assert
    expect(onError).toHaveBeenCalledTimes(1);
    expect(onAbort).toHaveBeenCalledTimes(0);
    expect(types.some((part) => part.type === 'error')).toBe(true);
  });
});

describe('aborts and retries', () => {
  const abortError = () =>
    new DOMException('The operation was aborted.', 'AbortError');
  const timeoutError = () =>
    new DOMException('The operation timed out.', 'TimeoutError');

  const withSignal = (abortSignal: AbortSignal) => ({
    ...MockLanguageModel.callOptions(),
    abortSignal,
  });

  describe('final error', () => {
    it('should throw the abort error, not a RetryError, when the call is cancelled after a failover', async () => {
      // Arrange
      const controller = new AbortController();
      const fallbackModel = MockLanguageModel.from({
        doGenerate: async () => {
          controller.abort();
          throw abortError();
        },
      });
      const model = createRetryableModel({
        model: MockLanguageModel.from(retryableError),
        retries: [fallbackModel],
      });

      // Act
      const result = model.doGenerate(withSignal(controller.signal));

      // Assert
      await expect(result).rejects.toMatchObject({ name: 'AbortError' });
      expect(fallbackModel.doGenerate).toHaveBeenCalledTimes(1);
    });

    it('should reject doStream with the abort error, not a RetryError, when the call is cancelled after a failover', async () => {
      // Arrange
      const controller = new AbortController();
      const fallbackModel = MockLanguageModel.from({
        doStream: async () => {
          controller.abort();
          throw abortError();
        },
      });
      const model = createRetryableModel({
        model: MockLanguageModel.from({
          doStream: async () => {
            throw retryableError;
          },
        }),
        retries: [fallbackModel],
      });

      // Act
      const result = model.doStream(withSignal(controller.signal));

      // Assert
      await expect(result).rejects.toMatchObject({ name: 'AbortError' });
      expect(fallbackModel.doStream).toHaveBeenCalledTimes(1);
    });

    it('should still throw a RetryError when a retry times out on its own deadline', async () => {
      // Arrange: the call itself was never aborted, so the timeout is a
      // failure of the retry like any other.
      const controller = new AbortController();
      const model = createRetryableModel({
        model: MockLanguageModel.from(retryableError),
        retries: [
          MockLanguageModel.from({
            doGenerate: async () => {
              throw timeoutError();
            },
          }),
        ],
      });

      // Act
      const result = model.doGenerate(withSignal(controller.signal));

      // Assert
      await expect(result).rejects.toThrow(RetryError);
    });
  });

  describe('retry against a cancelled call', () => {
    it('should not retry after a user cancel, even when the retry has a timeout', async () => {
      // Arrange
      const controller = new AbortController();
      const fallbackModel = MockLanguageModel.from(mockResultText);
      const onRetry = vi.fn<OnRetry>();
      const model = createRetryableModel({
        model: MockLanguageModel.from({
          doGenerate: async () => {
            controller.abort();
            throw abortError();
          },
        }),
        retries: [{ model: fallbackModel, timeout: 30_000 }],
        onRetry,
      });

      // Act
      const result = model.doGenerate(withSignal(controller.signal));

      // Assert
      await expect(result).rejects.toMatchObject({ name: 'AbortError' });
      expect(fallbackModel.doGenerate).toHaveBeenCalledTimes(0);
      expect(onRetry).toHaveBeenCalledTimes(0);
    });

    it('should still retry after the call timed out when the retry has a timeout', async () => {
      // Arrange
      const controller = new AbortController();
      const fallbackModel = MockLanguageModel.from(mockResultText);
      const model = createRetryableModel({
        model: MockLanguageModel.from({
          doGenerate: async () => {
            const error = timeoutError();
            controller.abort(error);
            throw error;
          },
        }),
        retries: [{ model: fallbackModel, timeout: 30_000 }],
      });

      // Act
      const result = await model.doGenerate(withSignal(controller.signal));

      // Assert
      expect(result.content).toEqual(mockResult.content);
      expect(fallbackModel.doGenerate).toHaveBeenCalledTimes(1);
    });
  });

  describe('retry timing out on its own deadline', () => {
    it('should throw a RetryError when the retry times out on its own deadline after the call deadline fired', async () => {
      // Arrange
      const controller = new AbortController();
      const model = createRetryableModel({
        model: MockLanguageModel.from({
          doGenerate: async () => {
            const error = timeoutError();
            controller.abort(error);
            throw error;
          },
        }),
        retries: [
          {
            model: MockLanguageModel.from({
              doGenerate: async () => {
                throw timeoutError();
              },
            }),
            timeout: 30_000,
          },
        ],
      });

      // Act
      const result = model.doGenerate(withSignal(controller.signal));

      // Assert
      await expect(result).rejects.toThrow(RetryError);
    });

    it('should forward an error part when a stream retry times out on its own deadline after the call deadline fired', async () => {
      // Arrange
      const controller = new AbortController();
      const model = createRetryableModel({
        model: MockLanguageModel.from({
          doStream: async () => {
            const error = timeoutError();
            controller.abort(error);
            throw error;
          },
        }),
        retries: [
          {
            model: MockLanguageModel.from({
              doStream: async () => ({
                stream: new ReadableStream<LanguageModelStreamPart>({
                  start(streamController) {
                    streamController.enqueue(Language.streamStart());
                    streamController.error(timeoutError());
                  },
                }),
              }),
            }),
            timeout: 30_000,
          },
        ],
      });

      // Act
      const { stream } = await model.doStream(withSignal(controller.signal));
      const parts = await Streams.toArray(stream);

      // Assert
      const errorPart = parts.find((part) => part.type === 'error');
      expect(RetryError.isInstance(errorPart?.error)).toBe(true);
    });
  });

  describe('result retry against a cancelled call', () => {
    it('should return the result instead of retrying when the call is already cancelled', async () => {
      // Arrange
      const controller = new AbortController();
      const fallbackModel = MockLanguageModel.from(mockResultText);
      const model = createRetryableModel({
        model: MockLanguageModel.from({
          doGenerate: async () => {
            controller.abort();
            return contentFilterResult;
          },
        }),
        retries: [
          (context) =>
            isResultAttempt(context.current)
              ? { model: fallbackModel, delay: 1_000 }
              : undefined,
        ],
      });

      // Act
      const result = await model.doGenerate(withSignal(controller.signal));

      // Assert
      expect(result.finishReason).toEqual(contentFilterResult.finishReason);
      expect(fallbackModel.doGenerate).toHaveBeenCalledTimes(0);
    });

    it('should reject without a phantom attempt when the call is cancelled during the delay before a finish retry', async () => {
      // Arrange
      const controller = new AbortController();
      const fallbackModel = MockLanguageModel.from({
        doStream: mockStreamChunks,
      });
      const onError = vi.fn<OnError>();
      const model = createRetryableModel({
        model: MockLanguageModel.from({ doStream: contentFilterStreamChunks }),
        retries: [
          (context) =>
            isResultAttempt(context.current)
              ? { model: fallbackModel, delay: 5_000 }
              : undefined,
        ],
        onError,
      });
      setTimeout(() => controller.abort(), 10);

      // Act
      const { stream } = await model.doStream(withSignal(controller.signal));
      const result = Streams.toArray(stream);

      // Assert
      await expect(result).rejects.toMatchObject({ name: 'AbortError' });
      expect(onError.mock.calls.length).toBe(0);
      expect(fallbackModel.doStream).toHaveBeenCalledTimes(0);
    });
  });

  describe('backoff delay', () => {
    it('should wait out the delay when the call timed out and the retry has its own deadline', async () => {
      // Arrange
      const controller = new AbortController();
      const fallbackModel = MockLanguageModel.from(mockResultText);
      const model = createRetryableModel({
        model: MockLanguageModel.from({
          doGenerate: async () => {
            const error = timeoutError();
            controller.abort(error);
            throw error;
          },
        }),
        retries: [{ model: fallbackModel, timeout: 30_000, delay: 20 }],
      });

      // Act
      const result = await model.doGenerate(withSignal(controller.signal));

      // Assert
      expect(result.content).toEqual(mockResult.content);
      expect(fallbackModel.doGenerate).toHaveBeenCalledTimes(1);
    });

    it('should wait out the delay before a stream retry when the call timed out and the retry has its own deadline', async () => {
      // Arrange
      const controller = new AbortController();
      const fallbackModel = MockLanguageModel.from({
        doStream: mockStreamChunks,
      });
      const model = createRetryableModel({
        model: MockLanguageModel.from({
          doStream: async () => ({
            stream: new ReadableStream<LanguageModelStreamPart>({
              start(streamController) {
                streamController.enqueue(Language.streamStart());
                const error = timeoutError();
                controller.abort(error);
                streamController.error(error);
              },
            }),
          }),
        }),
        retries: [{ model: fallbackModel, timeout: 30_000, delay: 20 }],
      });

      // Act
      const { stream } = await model.doStream(withSignal(controller.signal));
      const parts = await Streams.toArray(stream);

      // Assert
      expect(partsToText(parts)).toBe('Hello, world!');
      expect(fallbackModel.doStream).toHaveBeenCalledTimes(1);
    });

    it('should end the delay when the user cancels during it', async () => {
      // Arrange
      const controller = new AbortController();
      const fallbackModel = MockLanguageModel.from(mockResultText);
      const model = createRetryableModel({
        model: MockLanguageModel.from({
          doGenerate: async () => {
            setTimeout(() => controller.abort(), 10);
            throw retryableError;
          },
        }),
        retries: [{ model: fallbackModel, timeout: 30_000, delay: 5_000 }],
      });

      // Act
      const result = model.doGenerate(withSignal(controller.signal));

      // Assert
      await expect(result).rejects.toMatchObject({ name: 'AbortError' });
      expect(fallbackModel.doGenerate).toHaveBeenCalledTimes(0);
    });
  });
});

describe('onFailure and aborts', () => {
  const abortError = () =>
    new DOMException('The operation was aborted.', 'AbortError');

  const withSignal = (abortSignal?: AbortSignal) => ({
    ...MockLanguageModel.callOptions(),
    abortSignal,
  });

  const contentParts: Array<LanguageModelStreamPart> = [
    Language.streamStart(),
    { type: 'text-start', id: '0' },
    { type: 'text-delta', id: '0', delta: 'Hello' },
  ];

  /**
   * A model that streams content and then fails: `fail` runs once the content
   * is enqueued and decides how, from a rejection to an error part.
   */
  const failingAfterContent = (
    fail: (
      controller: ReadableStreamDefaultController<LanguageModelStreamPart>,
      abortSignal: AbortSignal | undefined,
    ) => void,
  ) =>
    MockLanguageModel.from({
      doStream: async (opts: LanguageModelCallOptions) => ({
        stream: new ReadableStream<LanguageModelStreamPart>({
          start(controller) {
            for (const part of contentParts) controller.enqueue(part);
            fail(controller, opts.abortSignal);
          },
        }),
      }),
    });

  /** Read a stream to its end, ignoring whether it rejects. */
  const drain = async (stream: ReadableStream<LanguageModelStreamPart>) => {
    try {
      for await (const _ of stream) {
      }
    } catch {}
  };

  describe('after content', () => {
    it('should call onFailure, not aborted, when the stream rejects with an error', async () => {
      // Arrange
      const error = new Error('connection reset');
      const baseModel = failingAfterContent((controller) =>
        controller.error(error),
      );
      const onFailure = vi.fn<OnFailure>();
      const onSuccess = vi.fn<OnSuccess>();
      const model = createRetryableModel({
        model: baseModel,
        retries: [],
        onFailure,
        onSuccess,
      });

      // Act
      const { stream } = await model.doStream(withSignal());
      await drain(stream);

      // Assert
      expect(onFailure.mock.calls.length).toBe(1);
      expect(onSuccess.mock.calls.length).toBe(0);
      const [failure] = onFailure.mock.calls[0]!;
      expect(failure.aborted).toBe(false);
      expect(failure.error).toBe(error);
      expect(failure.current.error).toBe(error);
      expect(failure.current.model).toBe(baseModel);
      expect(failure.attempts.at(-1)).toBe(failure.current);
      expect(failure.attempts.length).toBe(1);
    });

    it('should call onFailure, not aborted, when the stream carries an error part', async () => {
      // Arrange
      const error = new Error('overloaded');
      const baseModel = failingAfterContent((controller) => {
        controller.enqueue({ type: 'error', error });
        controller.close();
      });
      const onFailure = vi.fn<OnFailure>();
      const model = createRetryableModel({
        model: baseModel,
        retries: [],
        onFailure,
      });

      // Act
      const { stream } = await model.doStream(withSignal());
      await drain(stream);

      // Assert
      expect(onFailure.mock.calls.length).toBe(1);
      const [failure] = onFailure.mock.calls[0]!;
      expect(failure.aborted).toBe(false);
      expect(failure.error).toBe(error);
      expect(failure.current.error).toBe(error);
    });

    it('should call onFailure, aborted, when the call is aborted', async () => {
      // Arrange
      const controller = new AbortController();
      const baseModel = failingAfterContent((streamController, signal) =>
        signal?.addEventListener(
          'abort',
          () => streamController.error(abortError()),
          { once: true },
        ),
      );
      const onFailure = vi.fn<OnFailure>();
      const model = createRetryableModel({
        model: baseModel,
        retries: [],
        onFailure,
      });

      // Act
      const { stream } = await model.doStream(withSignal(controller.signal));
      const reading = drain(stream);
      controller.abort();
      await reading;

      // Assert
      expect(onFailure.mock.calls.length).toBe(1);
      const [failure] = onFailure.mock.calls[0]!;
      expect(failure.aborted).toBe(true);
      expect(failure.error).toMatchObject({ name: 'AbortError' });
    });

    it('should call onFailure, aborted, when the model sends an abort error part', async () => {
      // Arrange
      const controller = new AbortController();
      const baseModel = failingAfterContent((streamController, signal) =>
        signal?.addEventListener(
          'abort',
          () => {
            streamController.enqueue({ type: 'error', error: abortError() });
            streamController.close();
          },
          { once: true },
        ),
      );
      const onFailure = vi.fn<OnFailure>();
      const model = createRetryableModel({
        model: baseModel,
        retries: [],
        onFailure,
      });

      // Act
      const { stream } = await model.doStream(withSignal(controller.signal));
      const reading = drain(stream);
      controller.abort();
      await reading;

      // Assert
      expect(onFailure.mock.calls.length).toBe(1);
      expect(onFailure.mock.calls[0]![0].aborted).toBe(true);
    });
  });

  describe('cancel during the wait before a result retry', () => {
    it('should call onFailure, aborted, in doGenerate', async () => {
      // Arrange
      const controller = new AbortController();
      const baseModel = MockLanguageModel.from(contentFilterResult);
      const fallbackModel = MockLanguageModel.from(mockResultText);
      const onFailure = vi.fn<OnFailure>();
      const onSuccess = vi.fn<OnSuccess>();
      const onError = vi.fn<OnError>();
      const model = createRetryableModel({
        model: baseModel,
        retries: [
          (context) =>
            isResultAttempt(context.current)
              ? { model: fallbackModel, delay: 5_000 }
              : undefined,
        ],
        onFailure,
        onSuccess,
        onError,
      });
      setTimeout(() => controller.abort(), 10);

      // Act
      const result = model.doGenerate(withSignal(controller.signal));
      await expect(result).rejects.toMatchObject({ name: 'AbortError' });

      // Assert
      expect(onSuccess.mock.calls.length).toBe(0);
      expect(onFailure.mock.calls.length).toBe(1);
      const [failure] = onFailure.mock.calls[0]!;
      expect(failure.aborted).toBe(true);
      expect(failure.current.type).toBe('error');
      expect(failure.current.model).toBe(baseModel);
      expect(failure.attempts.map((attempt) => attempt.type)).toEqual([
        'result',
        'error',
      ]);
      expect(fallbackModel.doGenerate).toHaveBeenCalledTimes(0);
      expect(onError.mock.calls.length).toBe(0);
    });

    it('should call onFailure, aborted, before a finish retry in a stream', async () => {
      // Arrange
      const controller = new AbortController();
      const fallbackModel = MockLanguageModel.from({
        doStream: mockStreamChunks,
      });
      const onFailure = vi.fn<OnFailure>();
      const model = createRetryableModel({
        model: MockLanguageModel.from({ doStream: contentFilterStreamChunks }),
        retries: [
          (context) =>
            isResultAttempt(context.current)
              ? { model: fallbackModel, delay: 5_000 }
              : undefined,
        ],
        onFailure,
      });
      setTimeout(() => controller.abort(), 10);

      // Act
      const { stream } = await model.doStream(withSignal(controller.signal));
      await drain(stream);

      // Assert
      expect(onFailure.mock.calls.length).toBe(1);
      const [failure] = onFailure.mock.calls[0]!;
      expect(failure.aborted).toBe(true);
      expect(failure.attempts.map((attempt) => attempt.type)).toEqual([
        'result',
        'error',
      ]);
    });
  });

  describe('before content and in doGenerate', () => {
    it('should flag a stream error before content as not aborted', async () => {
      // Arrange
      const onFailure = vi.fn<OnFailure>();
      const model = createRetryableModel({
        model: MockLanguageModel.from({
          doStream: errorStreamChunks(nonRetryableError),
        }),
        retries: [],
        onFailure,
      });

      // Act
      const { stream } = await model.doStream(withSignal());
      await drain(stream);

      // Assert
      expect(onFailure.mock.calls.length).toBe(1);
      expect(onFailure.mock.calls[0]![0].aborted).toBe(false);
    });

    it('should flag a cancel before content as aborted', async () => {
      // Arrange
      const controller = new AbortController();
      const onFailure = vi.fn<OnFailure>();
      const model = createRetryableModel({
        model: MockLanguageModel.from({
          doStream: async () => {
            controller.abort();
            throw abortError();
          },
        }),
        retries: [],
        onFailure,
      });

      // Act
      const result = model.doStream(withSignal(controller.signal));
      await expect(result).rejects.toMatchObject({ name: 'AbortError' });

      // Assert
      expect(onFailure.mock.calls.length).toBe(1);
      expect(onFailure.mock.calls[0]![0].aborted).toBe(true);
    });

    it('should flag a cancel in doGenerate after a failover as aborted', async () => {
      // Arrange
      const controller = new AbortController();
      const onFailure = vi.fn<OnFailure>();
      const model = createRetryableModel({
        model: MockLanguageModel.from(retryableError),
        retries: [
          MockLanguageModel.from({
            doGenerate: async () => {
              controller.abort();
              throw abortError();
            },
          }),
        ],
        onFailure,
      });

      // Act
      const result = model.doGenerate(withSignal(controller.signal));
      await expect(result).rejects.toMatchObject({ name: 'AbortError' });

      // Assert
      expect(onFailure.mock.calls.length).toBe(1);
      const [failure] = onFailure.mock.calls[0]!;
      expect(failure.aborted).toBe(true);
      expect(failure.attempts.length).toBe(2);
    });

    it('should flag exhausted retries in doGenerate as not aborted', async () => {
      // Arrange
      const onFailure = vi.fn<OnFailure>();
      const model = createRetryableModel({
        model: MockLanguageModel.from(retryableError),
        retries: [MockLanguageModel.from(nonRetryableError)],
        onFailure,
      });

      // Act
      const result = model.doGenerate(withSignal());
      await expect(result).rejects.toThrow(RetryError);

      // Assert
      expect(onFailure.mock.calls.length).toBe(1);
      expect(onFailure.mock.calls[0]![0].aborted).toBe(false);
    });
  });
});
