import { APICallError } from 'ai';
import { describe, expect, it } from 'vitest';
import { findRetryModel } from './find-retry-model.js';
import { MockLanguageModel } from './test-utils.js';
import type {
  LanguageModel,
  LanguageModelCallOptions,
  ResolvableLanguageModel,
  Retry,
  Retryable,
  RetryContext,
} from './types.js';

const options: LanguageModelCallOptions = {
  prompt: [
    { role: 'user', content: [{ type: 'text', text: 'Hello, world!' }] },
  ],
};

describe('findRetryModel', () => {
  describe('with function retryables', () => {
    it('should find retry model from retryable function', async () => {
      const primaryModel = new MockLanguageModel();
      const fallbackModel = new MockLanguageModel();
      const error = new APICallError({
        message: 'Test error',
        statusCode: 500,
        url: '',
        requestBodyValues: {},
      });

      const retryable: Retryable<LanguageModel> = (context) => {
        if (context.current.type === 'error') {
          return { model: fallbackModel, maxAttempts: 1 };
        }
        return undefined;
      };

      const context: RetryContext<LanguageModel> = {
        current: { type: 'error', error, model: primaryModel, options },
        attempts: [{ type: 'error', error, model: primaryModel, options }],
      };

      const result = await findRetryModel([retryable], context);

      expect(result).toEqual({
        model: fallbackModel,
        maxAttempts: 1,
      });
    });

    it('should return undefined when retryable returns undefined', async () => {
      const primaryModel = new MockLanguageModel();
      const error = new Error('Test error');

      const retryable: Retryable<LanguageModel> = () => undefined;

      const context: RetryContext<LanguageModel> = {
        current: { type: 'error', error, model: primaryModel, options },
        attempts: [{ type: 'error', error, model: primaryModel, options }],
      };

      const result = await findRetryModel([retryable], context);

      expect(result).toBeUndefined();
    });

    it('should handle async retryable functions', async () => {
      const primaryModel = new MockLanguageModel();
      const fallbackModel = new MockLanguageModel();
      const error = new Error('Test error');

      const retryable: Retryable<LanguageModel> = async () => {
        await new Promise((resolve) => setTimeout(resolve, 10));
        return { model: fallbackModel, maxAttempts: 2 };
      };

      const context: RetryContext<LanguageModel> = {
        current: { type: 'error', error, model: primaryModel, options },
        attempts: [{ type: 'error', error, model: primaryModel, options }],
      };

      const result = await findRetryModel([retryable], context);

      expect(result).toEqual({
        model: fallbackModel,
        maxAttempts: 2,
      });
    });
  });

  describe('with static Retry objects', () => {
    it('should find retry model from static Retry object', async () => {
      const primaryModel = new MockLanguageModel();
      const fallbackModel = new MockLanguageModel();
      const error = new Error('Test error');

      const retry: Retry<LanguageModel> = {
        model: fallbackModel,
        maxAttempts: 3,
        delay: 1000,
      };

      const context: RetryContext<LanguageModel> = {
        current: { type: 'error', error, model: primaryModel, options },
        attempts: [{ type: 'error', error, model: primaryModel, options }],
      };

      const result = await findRetryModel([retry], context);

      expect(result).toEqual({
        model: fallbackModel,
        maxAttempts: 3,
        delay: 1000,
      });
    });

    it('should skip static Retry for result-based attempts', async () => {
      const primaryModel = new MockLanguageModel();
      const fallbackModel = new MockLanguageModel();

      const retry: Retry<LanguageModel> = {
        model: fallbackModel,
        maxAttempts: 1,
      };

      const context: RetryContext<LanguageModel> = {
        current: {
          type: 'result',
          result: {
            content: [],
            finishReason: 'content-filter',
            usage: {
              inputTokens: {
                total: 0,
                noCache: 0,
                cacheRead: 0,
                cacheWrite: 0,
              },
              outputTokens: { total: 0, text: 0, reasoning: 0 },
            },
            warnings: [],
          },
          model: primaryModel,
          options: options,
        },
        attempts: [],
      };

      const result = await findRetryModel([retry], context);

      expect(result).toBeUndefined();
    });
  });

  describe('with plain models', () => {
    it('should find retry model from plain model', async () => {
      const primaryModel = new MockLanguageModel();
      const fallbackModel = new MockLanguageModel();
      const error = new Error('Test error');

      const context: RetryContext<LanguageModel> = {
        current: { type: 'error', error, model: primaryModel, options },
        attempts: [{ type: 'error', error, model: primaryModel, options }],
      };

      const result = await findRetryModel([fallbackModel], context);

      expect(result).toEqual({
        model: fallbackModel,
      });
    });

    it('should skip plain models for result-based attempts', async () => {
      const primaryModel = new MockLanguageModel();
      const fallbackModel = new MockLanguageModel();

      const context: RetryContext<LanguageModel> = {
        current: {
          type: 'result',
          result: {
            content: [],
            finishReason: 'content-filter',
            usage: {
              inputTokens: {
                total: 0,
                noCache: 0,
                cacheRead: 0,
                cacheWrite: 0,
              },
              outputTokens: { total: 0, text: 0, reasoning: 0 },
            },
            warnings: [],
          },
          model: primaryModel,
          options: options,
        },
        attempts: [],
      };

      const result = await findRetryModel([fallbackModel], context);

      expect(result).toBeUndefined();
    });
  });

  describe('maxAttempts logic', () => {
    it('should respect maxAttempts limit', async () => {
      const primaryModel = new MockLanguageModel();
      const fallbackModel = new MockLanguageModel();
      const error = new Error('Test error');

      const retry: Retry<LanguageModel> = {
        model: fallbackModel,
        maxAttempts: 2,
      };

      const context: RetryContext<LanguageModel> = {
        current: { type: 'error', error, model: primaryModel, options },
        attempts: [
          { type: 'error', error, model: fallbackModel, options: options },
          { type: 'error', error, model: fallbackModel, options: options },
        ],
      };

      const result = await findRetryModel([retry], context);

      expect(result).toBeUndefined();
    });

    it('should allow retry when under maxAttempts', async () => {
      const primaryModel = new MockLanguageModel();
      const fallbackModel = new MockLanguageModel();
      const error = new Error('Test error');

      const retry: Retry<LanguageModel> = {
        model: fallbackModel,
        maxAttempts: 3,
      };

      const context: RetryContext<LanguageModel> = {
        current: {
          type: 'error',
          error,
          model: primaryModel,
          options: options,
        },
        attempts: [
          { type: 'error', error, model: primaryModel, options: options },
          { type: 'error', error, model: fallbackModel, options: options },
        ],
      };

      const result = await findRetryModel([retry], context);

      expect(result).toEqual({
        model: fallbackModel,
        maxAttempts: 3,
      });
    });

    it('should default maxAttempts to 1', async () => {
      const primaryModel = new MockLanguageModel();
      const fallbackModel = new MockLanguageModel();
      const error = new Error('Test error');

      const retry: Retry<LanguageModel> = {
        model: fallbackModel,
      };

      const context: RetryContext<LanguageModel> = {
        current: { type: 'error', error, model: primaryModel, options },
        attempts: [{ type: 'error', error, model: fallbackModel, options }],
      };

      const result = await findRetryModel([retry], context);

      expect(result).toBeUndefined();
    });
  });

  describe('with multiple retries', () => {
    it('should return first matching retry', async () => {
      const primaryModel = new MockLanguageModel();
      const fallback1 = new MockLanguageModel();
      const fallback2 = new MockLanguageModel();
      const error = new APICallError({
        message: 'Test error',
        statusCode: 503,
        url: '',
        requestBodyValues: {},
      });

      const retryable1: Retryable<LanguageModel> = () => undefined;
      const retryable2: Retryable<LanguageModel> = () => ({
        model: fallback1,
        maxAttempts: 1,
      });
      const retryable3: Retryable<LanguageModel> = () => ({
        model: fallback2,
        maxAttempts: 1,
      });

      const context: RetryContext<LanguageModel> = {
        current: {
          type: 'error',
          error,
          model: primaryModel,
          options: options,
        },
        attempts: [
          { type: 'error', error, model: primaryModel, options: options },
        ],
      };

      const result = await findRetryModel(
        [retryable1, retryable2, retryable3],
        context,
      );

      expect(result?.model).toBe(fallback1);
    });

    it('should skip exhausted retries and find next available', async () => {
      const primaryModel = new MockLanguageModel();
      const fallback1 = new MockLanguageModel();
      const fallback2 = new MockLanguageModel();
      const error = new Error('Test error');

      const retry1: Retry<LanguageModel> = {
        model: fallback1,
        maxAttempts: 1,
      };
      const retry2: Retry<LanguageModel> = {
        model: fallback2,
        maxAttempts: 1,
      };

      const context: RetryContext<LanguageModel> = {
        current: {
          type: 'error',
          error,
          model: primaryModel,
          options: options,
        },
        attempts: [
          { type: 'error', error, model: primaryModel, options: options },
          { type: 'error', error, model: fallback1, options: options },
        ],
      };

      const result = await findRetryModel([retry1, retry2], context);

      expect(result?.model).toBe(fallback2);
    });
  });

  describe('model resolution', () => {
    it('should resolve string model IDs to gateway provider', async () => {
      const primaryModel = new MockLanguageModel();
      const error = new Error('Test error');

      const context: RetryContext<LanguageModel> = {
        current: {
          type: 'error',
          error,
          model: primaryModel,
          options: options,
        },
        attempts: [
          { type: 'error', error, model: primaryModel, options: options },
        ],
      };

      const result = await findRetryModel(['openai/gpt-4o'], context);

      expect(result).toBeDefined();
      expect(result?.model.provider).toBe('gateway');
      expect(result?.model.modelId).toBe('openai/gpt-4o');
    });

    it('should preserve retry options when resolving models', async () => {
      const primaryModel = new MockLanguageModel();
      const error = new Error('Test error');

      const retry: Retry<ResolvableLanguageModel> = {
        model: 'openai/gpt-3.5-turbo',
        maxAttempts: 5,
        delay: 2000,
        backoffFactor: 2,
        timeout: 30000,
      };

      const context: RetryContext<LanguageModel> = {
        current: {
          type: 'error',
          error,
          model: primaryModel,
          options: options,
        },
        attempts: [
          { type: 'error', error, model: primaryModel, options: options },
        ],
      };

      const result = await findRetryModel([retry], context);

      expect(result).toEqual({
        model: expect.any(Object),
        maxAttempts: 5,
        delay: 2000,
        backoffFactor: 2,
        timeout: 30000,
      });
    });
  });
});
