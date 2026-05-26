import {
  context,
  SpanStatusCode,
  trace,
  type Tracer,
} from '@opentelemetry/api';
import { AsyncLocalStorageContextManager } from '@opentelemetry/context-async-hooks';
import type { InMemorySpanExporter } from '@opentelemetry/sdk-trace-base';
import { APICallError } from 'ai';
import { beforeEach, describe, expect, it } from 'vitest';
import { createRetryable } from '../create-retryable-model.js';
import { isErrorAttempt } from './guards.js';
import {
  attemptSpans,
  createSpanExporter,
  drainStream,
  embeddingCallOptions,
  errorStreamChunks,
  findSpan,
  generateTextResult,
  imageCallOptions,
  languageCallOptions,
  mockEmbeddings,
  MockEmbeddingModel,
  mockImageResult,
  MockImageModel,
  MockLanguageModel,
  mockStream,
  retryableError,
  successStreamChunks,
} from './test-utils.js';
import type {
  EmbeddingModel,
  ImageModel,
  LanguageModel,
  Retryable,
} from '../types.js';

let exporter: InMemorySpanExporter;
let tracer: Tracer;

beforeEach(() => {
  ({ exporter, tracer } = createSpanExporter());
});

describe('telemetry', () => {
  describe('generateText', () => {
    it('should not emit spans when telemetry is not configured', async () => {
      // Arrange
      const baseModel = new MockLanguageModel({
        doGenerate: generateTextResult('ok'),
      });
      const model = createRetryable({ model: baseModel, retries: [] });

      // Act
      await model.doGenerate(languageCallOptions);

      // Assert
      expect(exporter.getFinishedSpans().length).toBe(0);
    });

    it('should not emit spans when telemetry is disabled', async () => {
      // Arrange
      const baseModel = new MockLanguageModel({
        doGenerate: generateTextResult('ok'),
      });
      const model = createRetryable({
        model: baseModel,
        retries: [],
        experimental_telemetry: { isEnabled: false, tracer },
      });

      // Act
      await model.doGenerate(languageCallOptions);

      // Assert
      expect(exporter.getFinishedSpans().length).toBe(0);
    });

    it('should emit an operation span and one attempt span on success', async () => {
      // Arrange
      const baseModel = new MockLanguageModel({
        doGenerate: generateTextResult('ok'),
      });
      const model = createRetryable({
        model: baseModel,
        retries: [],
        experimental_telemetry: { isEnabled: true, tracer },
      });

      // Act
      await model.doGenerate(languageCallOptions);

      // Assert
      const operation = findSpan(exporter, 'ai_retry.doGenerate');
      const attempts = attemptSpans(exporter);
      expect(attempts.length).toBe(1);

      expect(operation.attributes['ai_retry.operation']).toBe('doGenerate');
      expect(operation.attributes['gen_ai.operation.name']).toBe('chat');
      expect(operation.attributes['ai_retry.outcome']).toBe('success');
      expect(operation.attributes['ai_retry.attempts']).toBe(1);
      expect(operation.attributes['ai_retry.model.start']).toBe(
        `mock-provider/${baseModel.modelId}`,
      );
      expect(operation.attributes['ai_retry.model.final']).toBe(
        `mock-provider/${baseModel.modelId}`,
      );

      expect(attempts[0]!.attributes['ai_retry.attempt.number']).toBe(1);
      expect(attempts[0]!.attributes['ai_retry.attempt.outcome']).toBe(
        'success',
      );
      expect(attempts[0]!.attributes['ai_retry.attempt.type']).toBe('result');
      expect(attempts[0]!.attributes['ai_retry.attempt.model']).toBe(
        `mock-provider/${baseModel.modelId}`,
      );
      expect(attempts[0]!.attributes['ai_retry.attempt.finish_reason']).toBe(
        'stop',
      );

      /**
       * Standard gen_ai attributes are kept alongside the ai_retry.* ones so
       * backends like Langfuse export and render attempt spans by default.
       */
      expect(attempts[0]!.attributes['gen_ai.request.model']).toBe(
        baseModel.modelId,
      );

      /** The attempt span nests under the operation span. */
      expect(attempts[0]!.parentSpanContext?.spanId).toBe(
        operation.spanContext().spanId,
      );
    });

    it('should record the failed attempt and the successful fallback', async () => {
      // Arrange
      const baseModel = new MockLanguageModel({ doGenerate: retryableError });
      const fallbackModel = new MockLanguageModel({
        doGenerate: generateTextResult('ok'),
      });
      const fallback: Retryable<LanguageModel> = (ctx) =>
        isErrorAttempt(ctx.current)
          ? { model: fallbackModel, maxAttempts: 1 }
          : undefined;
      const model = createRetryable({
        model: baseModel,
        retries: [fallback],
        experimental_telemetry: { isEnabled: true, tracer },
      });

      // Act
      await model.doGenerate(languageCallOptions);

      // Assert
      const attempts = attemptSpans(exporter);
      expect(attempts.length).toBe(2);

      expect(attempts[0]!.attributes['ai_retry.attempt.outcome']).toBe('retry');
      expect(attempts[0]!.attributes['ai_retry.attempt.type']).toBe('error');
      expect(attempts[0]!.attributes['ai_retry.attempt.model']).toBe(
        `mock-provider/${baseModel.modelId}`,
      );
      expect(attempts[0]!.attributes['ai_retry.attempt.error.name']).toBe(
        retryableError.name,
      );
      expect(attempts[0]!.attributes['ai_retry.attempt.error.message']).toBe(
        retryableError.message,
      );
      expect(attempts[0]!.attributes['ai_retry.attempt.error.status']).toBe(
        429,
      );
      expect(attempts[0]!.status.code).toBe(SpanStatusCode.ERROR);

      expect(attempts[1]!.attributes['ai_retry.attempt.outcome']).toBe(
        'success',
      );
      expect(attempts[1]!.attributes['ai_retry.attempt.type']).toBe('result');
      expect(attempts[1]!.attributes['ai_retry.attempt.model']).toBe(
        `mock-provider/${fallbackModel.modelId}`,
      );
      expect(attempts[1]!.status.code).toBe(SpanStatusCode.UNSET);

      const operation = findSpan(exporter, 'ai_retry.doGenerate');
      expect(operation.attributes['ai_retry.outcome']).toBe('success');
      expect(operation.attributes['ai_retry.attempts']).toBe(2);
      expect(operation.attributes['ai_retry.model.final']).toBe(
        `mock-provider/${fallbackModel.modelId}`,
      );
      expect(operation.status.code).toBe(SpanStatusCode.UNSET);
    });

    it('should record the error name, message, and cause on the attempt', async () => {
      // Arrange
      const cause = new Error('underlying socket failure');
      cause.name = 'SocketError';
      const wrappedError = new APICallError({
        message: 'upstream request failed',
        url: '',
        requestBodyValues: {},
        statusCode: 500,
        isRetryable: true,
        cause,
      });
      const baseModel = new MockLanguageModel({ doGenerate: wrappedError });
      const fallbackModel = new MockLanguageModel({
        doGenerate: generateTextResult('ok'),
      });
      const fallback: Retryable<LanguageModel> = (ctx) =>
        isErrorAttempt(ctx.current)
          ? { model: fallbackModel, maxAttempts: 1 }
          : undefined;
      const model = createRetryable({
        model: baseModel,
        retries: [fallback],
        experimental_telemetry: { isEnabled: true, tracer },
      });

      // Act
      await model.doGenerate(languageCallOptions);

      // Assert
      const attempt = attemptSpans(exporter)[0]!;
      expect(attempt.attributes['ai_retry.attempt.error.name']).toBe(
        wrappedError.name,
      );
      expect(attempt.attributes['ai_retry.attempt.error.message']).toBe(
        'upstream request failed',
      );
      expect(attempt.attributes['ai_retry.attempt.error.cause.name']).toBe(
        'SocketError',
      );
      expect(attempt.attributes['ai_retry.attempt.error.cause.message']).toBe(
        'underlying socket failure',
      );
    });

    it('should mark the operation and final attempt as failed when all attempts fail', async () => {
      // Arrange
      const baseModel = new MockLanguageModel({ doGenerate: retryableError });
      const fallbackModel = new MockLanguageModel({
        doGenerate: retryableError,
      });
      const fallback: Retryable<LanguageModel> = (ctx) =>
        isErrorAttempt(ctx.current) &&
        APICallError.isInstance(ctx.current.error)
          ? { model: fallbackModel, maxAttempts: 1 }
          : undefined;
      const model = createRetryable({
        model: baseModel,
        retries: [fallback],
        experimental_telemetry: { isEnabled: true, tracer },
      });

      // Act
      const result = model.doGenerate(languageCallOptions);

      // Assert
      await expect(result).rejects.toThrow();

      const attempts = attemptSpans(exporter);
      expect(attempts.length).toBe(2);
      expect(attempts[0]!.attributes['ai_retry.attempt.outcome']).toBe('retry');
      expect(attempts[1]!.attributes['ai_retry.attempt.outcome']).toBe(
        'failure',
      );
      expect(attempts[1]!.status.code).toBe(SpanStatusCode.ERROR);

      const operation = findSpan(exporter, 'ai_retry.doGenerate');
      expect(operation.attributes['ai_retry.outcome']).toBe('failure');
      expect(operation.attributes['ai_retry.attempts']).toBe(2);
      expect(operation.status.code).toBe(SpanStatusCode.ERROR);
    });

    it('should record the retry timeout on the attempt that runs under it', async () => {
      // Arrange
      const baseModel = new MockLanguageModel({ doGenerate: retryableError });
      const fallbackModel = new MockLanguageModel({
        doGenerate: generateTextResult('ok'),
      });
      const fallback: Retryable<LanguageModel> = (ctx) =>
        isErrorAttempt(ctx.current)
          ? { model: fallbackModel, maxAttempts: 1, timeout: 5000 }
          : undefined;
      const model = createRetryable({
        model: baseModel,
        retries: [fallback],
        experimental_telemetry: { isEnabled: true, tracer },
      });

      // Act
      await model.doGenerate(languageCallOptions);

      // Assert
      const attempts = attemptSpans(exporter);
      /** The first attempt ran without a retry-imposed timeout. */
      expect(
        attempts[0]!.attributes['ai_retry.attempt.timeout_ms'],
      ).toBeUndefined();
      /** The fallback attempt ran under the 5s timeout from the retry config. */
      expect(attempts[1]!.attributes['ai_retry.attempt.timeout_ms']).toBe(5000);
    });

    it('should record functionId and metadata on the operation span', async () => {
      // Arrange
      const baseModel = new MockLanguageModel({
        doGenerate: generateTextResult('ok'),
      });
      const model = createRetryable({
        model: baseModel,
        retries: [],
        experimental_telemetry: {
          isEnabled: true,
          tracer,
          functionId: 'my-fn',
          metadata: { env: 'test' },
        },
      });

      // Act
      await model.doGenerate(languageCallOptions);

      // Assert
      const operation = findSpan(exporter, 'ai_retry.doGenerate');
      expect(operation.attributes['ai_retry.function.id']).toBe('my-fn');
      expect(operation.attributes['ai_retry.metadata.env']).toBe('test');
    });

    it('should nest the operation span under a surrounding active span', async () => {
      // Arrange
      const contextManager = new AsyncLocalStorageContextManager();
      contextManager.enable();
      context.setGlobalContextManager(contextManager);

      try {
        const baseModel = new MockLanguageModel({
          doGenerate: generateTextResult('ok'),
        });
        const model = createRetryable({
          model: baseModel,
          retries: [],
          experimental_telemetry: { isEnabled: true, tracer },
        });
        const outer = tracer.startSpan('outer');

        // Act
        await context.with(trace.setSpan(context.active(), outer), async () => {
          await model.doGenerate(languageCallOptions);
        });
        outer.end();

        // Assert
        const operation = findSpan(exporter, 'ai_retry.doGenerate');
        const outerSpan = findSpan(exporter, 'outer');
        expect(operation.parentSpanContext?.spanId).toBe(
          outerSpan.spanContext().spanId,
        );
      } finally {
        context.disable();
      }
    });
  });

  describe('streamText', () => {
    it('should emit a doStream operation and attempt span on a successful stream', async () => {
      // Arrange
      const baseModel = new MockLanguageModel({
        doStream: mockStream(successStreamChunks('hi')),
      });
      const model = createRetryable({
        model: baseModel,
        retries: [],
        experimental_telemetry: { isEnabled: true, tracer },
      });

      // Act
      const { stream } = await model.doStream(languageCallOptions);
      await drainStream(stream);

      // Assert
      const operation = findSpan(exporter, 'ai_retry.doStream');
      const attempts = attemptSpans(exporter);
      expect(operation.attributes['ai_retry.operation']).toBe('doStream');
      expect(operation.attributes['gen_ai.operation.name']).toBe('chat');
      expect(operation.attributes['ai_retry.outcome']).toBe('success');
      expect(operation.attributes['ai_retry.attempts']).toBe(1);
      expect(attempts.length).toBe(1);
      expect(attempts[0]!.attributes['ai_retry.attempt.outcome']).toBe(
        'success',
      );
      expect(attempts[0]!.attributes['ai_retry.attempt.type']).toBe('result');
      expect(attempts[0]!.attributes['ai_retry.attempt.finish_reason']).toBe(
        'stop',
      );
    });

    it('should record a mid-stream error retry across two attempt spans', async () => {
      // Arrange
      const baseModel = new MockLanguageModel({
        doStream: mockStream(errorStreamChunks(retryableError)),
      });
      const fallbackModel = new MockLanguageModel({
        doStream: mockStream(successStreamChunks('hi')),
      });
      const fallback: Retryable<LanguageModel> = (ctx) =>
        isErrorAttempt(ctx.current)
          ? { model: fallbackModel, maxAttempts: 1 }
          : undefined;
      const model = createRetryable({
        model: baseModel,
        retries: [fallback],
        experimental_telemetry: { isEnabled: true, tracer },
      });

      // Act
      const { stream } = await model.doStream(languageCallOptions);
      await drainStream(stream);

      // Assert
      const attempts = attemptSpans(exporter);
      expect(attempts.length).toBe(2);
      expect(attempts[0]!.attributes['ai_retry.attempt.outcome']).toBe('retry');
      expect(attempts[0]!.attributes['ai_retry.attempt.type']).toBe('error');
      expect(attempts[0]!.attributes['ai_retry.attempt.error.name']).toBe(
        retryableError.name,
      );
      expect(attempts[1]!.attributes['ai_retry.attempt.outcome']).toBe(
        'success',
      );
      expect(attempts[1]!.attributes['ai_retry.attempt.type']).toBe('result');

      const operation = findSpan(exporter, 'ai_retry.doStream');
      expect(operation.attributes['ai_retry.outcome']).toBe('success');
      expect(operation.attributes['ai_retry.attempts']).toBe(2);
      expect(operation.attributes['ai_retry.model.final']).toBe(
        `mock-provider/${fallbackModel.modelId}`,
      );
    });
  });

  describe('embed', () => {
    it('should record telemetry for embedding retries', async () => {
      // Arrange
      const baseModel = new MockEmbeddingModel({ doEmbed: retryableError });
      const fallbackModel = new MockEmbeddingModel({ doEmbed: mockEmbeddings });
      const fallback: Retryable<EmbeddingModel> = (ctx) =>
        isErrorAttempt(ctx.current)
          ? { model: fallbackModel, maxAttempts: 1 }
          : undefined;
      const model = createRetryable({
        model: baseModel,
        retries: [fallback],
        experimental_telemetry: { isEnabled: true, tracer },
      });

      // Act
      await model.doEmbed(embeddingCallOptions);

      // Assert
      const operation = findSpan(exporter, 'ai_retry.doEmbed');
      const attempts = attemptSpans(exporter);
      expect(operation.attributes['gen_ai.operation.name']).toBe('embeddings');
      expect(operation.attributes['ai_retry.attempts']).toBe(2);
      expect(attempts.length).toBe(2);
      expect(attempts[0]!.attributes['ai_retry.attempt.outcome']).toBe('retry');
      expect(attempts[1]!.attributes['ai_retry.attempt.outcome']).toBe(
        'success',
      );
    });
  });

  describe('generateImage', () => {
    it('should record telemetry for image generation retries', async () => {
      // Arrange
      const baseModel = new MockImageModel({ doGenerate: retryableError });
      const fallbackModel = new MockImageModel({ doGenerate: mockImageResult });
      const fallback: Retryable<ImageModel> = (ctx) =>
        isErrorAttempt(ctx.current)
          ? { model: fallbackModel, maxAttempts: 1 }
          : undefined;
      const model = createRetryable({
        model: baseModel,
        retries: [fallback],
        experimental_telemetry: { isEnabled: true, tracer },
      });

      // Act
      await model.doGenerate(imageCallOptions);

      // Assert
      const operation = findSpan(exporter, 'ai_retry.doGenerate');
      const attempts = attemptSpans(exporter);
      expect(operation.attributes['gen_ai.operation.name']).toBe(
        'generate_content',
      );
      expect(operation.attributes['ai_retry.outcome']).toBe('success');
      expect(operation.attributes['ai_retry.attempts']).toBe(2);
      expect(attempts.length).toBe(2);
      expect(attempts[0]!.attributes['ai_retry.attempt.outcome']).toBe('retry');
      expect(attempts[0]!.attributes['ai_retry.attempt.type']).toBe('error');
      expect(attempts[1]!.attributes['ai_retry.attempt.outcome']).toBe(
        'success',
      );
    });
  });
});
