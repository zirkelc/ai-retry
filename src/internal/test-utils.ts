import type { Tracer } from '@opentelemetry/api';
import {
  BasicTracerProvider,
  InMemorySpanExporter,
  type ReadableSpan,
  SimpleSpanProcessor,
} from '@opentelemetry/sdk-trace-base';
import { type TextStreamPart } from 'ai';
import { Errors, Streams } from 'ai-test-kit';
import { Embedding, MockEmbeddingModel } from 'ai-test-kit/embedding';
import { Image, MockImageModel, validBase64Image } from 'ai-test-kit/image';
import { Language, MockLanguageModel, Options } from 'ai-test-kit/language';
import type {
  EmbeddingModel,
  EmbeddingModelEmbed,
  ImageModel,
  ImageModelCallOptions,
  ImageModelGenerate,
  LanguageModel,
  LanguageModelCallOptions,
  LanguageModelGenerate,
  LanguageModelResult,
  LanguageModelStreamPart,
  RetryContext,
  RetryErrorAttempt,
  RetryResultAttempt,
} from '../types.js';

/**
 * Re-exported for test convenience: the auto-detecting factory builds a
 * retryable model for any family from a single import.
 */
export { createRetryableModel } from './create-retryable-model.js';

/**
 * Re-exported ai-test-kit builders and mock factories. Tests construct mocks
 * with `MockLanguageModel.from(...)` / `MockEmbeddingModel.from(...)` /
 * `MockImageModel.from(...)`; the names are exported as both values and types.
 */
export {
  Embedding,
  Errors,
  Image,
  Language,
  MockEmbeddingModel,
  MockImageModel,
  MockLanguageModel,
  Options,
  Streams,
  validBase64Image,
};

export type LanguageModelGenerateFn = LanguageModel['doGenerate'];
export type LanguageModelStreamFn = LanguageModel['doStream'];
export type EmbeddingModelEmbedFn = EmbeddingModel['doEmbed'];
export type ImageModelGenerateFn = ImageModel['doGenerate'];

export const chunksToText = (chunks: TextStreamPart<any>[]): string => {
  return chunks
    .map((chunk) => (chunk.type === 'text-delta' ? chunk.text : ''))
    .join('');
};

export const finishReason = (
  chunks: TextStreamPart<any>[],
): string | undefined => {
  return chunks.find((chunk) => chunk.type === 'finish')?.finishReason;
};

export const errorFromChunks = (
  chunks: TextStreamPart<any>[],
): unknown | null => {
  const errorChunk = chunks.find((chunk) => chunk.type === 'error');
  return errorChunk && errorChunk.type === 'error' ? errorChunk.error : null;
};

/**
 * Build a synthetic `RetryContext` carrying an error attempt for a language
 * model. Used by unit tests that exercise condition predicates without
 * spinning up `generateText`.
 */
export const buildErrorContext = (
  error: unknown,
  model: MockLanguageModel = MockLanguageModel.from(),
): RetryContext<MockLanguageModel> => {
  const attempt: RetryErrorAttempt<MockLanguageModel> = {
    type: 'error',
    error,
    model,
    options: {} as LanguageModelCallOptions,
  };
  return { current: attempt, attempts: [attempt] };
};

/**
 * Build a synthetic `RetryContext` carrying a result attempt for a language
 * model.
 */
export const buildResultContext = (
  result: LanguageModelGenerate = Language.result([]),
  model: MockLanguageModel = MockLanguageModel.from(),
): RetryContext<MockLanguageModel> => {
  const attempt: RetryResultAttempt = {
    type: 'result',
    result,
    model,
    options: {} as LanguageModelCallOptions,
  };
  return { current: attempt, attempts: [attempt] };
};

/**
 * Build a synthetic `RetryContext` carrying an error attempt for an image
 * model.
 */
export const buildImageErrorContext = (
  error: unknown,
  model: MockImageModel = MockImageModel.from(),
): RetryContext<MockImageModel> => {
  const attempt: RetryErrorAttempt<MockImageModel> = {
    type: 'error',
    error,
    model,
    options: {} as ImageModelCallOptions,
  };
  return { current: attempt, attempts: [attempt] };
};

/**
 * Shared error constants — kept (rather than inlined to `Errors.*` at call
 * sites) because several tests assert identity, e.g.
 * `expect(ctx.current.error).toBe(retryableError)` and
 * `retryError.errors[0]).toBe(nonRetryableError)`. A fresh `Errors.rateLimited()`
 * per call site would not be the same reference.
 */
export const retryableError = Errors.rateLimited();
export const nonRetryableError = Errors.unauthorized();

export const mockResultText = 'Hello, world!';

/** A successful generate result with text content. */
export const mockResult: LanguageModelResult = Language.result(mockResultText);

/** A generate result that finished due to content filtering. */
export const contentFilterResult: LanguageModelResult = Language.result([], {
  finishReason: 'content-filter',
});

/** Stream parts for a full successful stream, including response metadata. */
export const mockStreamChunks: Array<LanguageModelStreamPart> = [
  Language.streamStart(),
  Language.streamResponseMetadata({
    id: 'id-0',
    modelId: 'mock-model-id',
    timestamp: new Date(0),
  }),
  ...Language.streamText(['Hello', ', ', 'world!'], { id: '1' }),
  Language.streamFinish(),
];

/** A successful embedding result. */
export const mockEmbeddings: EmbeddingModelEmbed = Embedding.result(
  [Embedding.vector(3)],
  { usage: Embedding.usage(5) },
);

/** A successful image generation result. */
export const mockImageResult: ImageModelGenerate = Image.result([
  validBase64Image,
]);

/** Stream parts for a successful stream: content then a `stop` finish. */
export const successStreamChunks = (
  text: string,
): Array<LanguageModelStreamPart> => [
  Language.streamStart(),
  ...Language.streamText(text, { id: '1' }),
  Language.streamFinish(),
];

/** Stream parts that error before any content is emitted. */
export const errorStreamChunks = (
  error: unknown,
): Array<LanguageModelStreamPart> => [
  Language.streamStart(),
  Language.streamError(error),
];

/** Concatenate the text deltas (raw parts carry `delta`, not `text`). */
export const partsToText = (parts: Array<LanguageModelStreamPart>): string =>
  parts.map((p) => (p.type === 'text-delta' ? p.delta : '')).join('');

/** Create an in-memory OpenTelemetry exporter and a tracer wired to it. */
export const createSpanExporter = (): {
  exporter: InMemorySpanExporter;
  tracer: Tracer;
} => {
  const exporter = new InMemorySpanExporter();
  const provider = new BasicTracerProvider({
    spanProcessors: [new SimpleSpanProcessor(exporter)],
  });
  return { exporter, tracer: provider.getTracer('test') };
};

/** The single finished span with the given name. Throws if it is missing. */
export const findSpan = (
  exporter: InMemorySpanExporter,
  name: string,
): ReadableSpan => {
  const found = exporter.getFinishedSpans().find((s) => s.name === name);
  if (!found) throw new Error(`expected a span named "${name}"`);
  return found;
};

/** Finished `ai_retry.attempt` spans, sorted by their 1-based number. */
export const attemptSpans = (
  exporter: InMemorySpanExporter,
): Array<ReadableSpan> =>
  exporter
    .getFinishedSpans()
    .filter((s) => s.name === 'ai_retry.attempt')
    .sort(
      (a, b) =>
        Number(a.attributes['ai_retry.attempt.number']) -
        Number(b.attributes['ai_retry.attempt.number']),
    );
