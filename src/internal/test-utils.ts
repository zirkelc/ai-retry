import type { Tracer } from '@opentelemetry/api';
import {
  BasicTracerProvider,
  InMemorySpanExporter,
  type ReadableSpan,
  SimpleSpanProcessor,
} from '@opentelemetry/sdk-trace-base';
import {
  APICallError,
  embed,
  embedMany,
  generateImage,
  generateText,
  type TextStreamPart,
} from 'ai';
import { Errors, Streams } from 'ai-test-kit';
import { Embedding, MockEmbeddingModel } from 'ai-test-kit/embedding';
import { Image, MockImageModel } from 'ai-test-kit/image';
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
  ModelRetryContext,
  ModelRetryErrorAttempt,
  ModelRetryResultAttempt,
} from '../types.js';
import type {
  CallArgs,
  CallFinishReason,
  CallResult,
  CallRetryContext,
  CallRetryErrorAttempt,
  CallRetryResultAttempt,
  EmbedManyResultInfo,
  EmbedResultInfo,
  GenerateImageResultInfo,
  GenerateTextResultInfo,
  StreamTextResultInfo,
} from '../call/types.js';
import { tagResult } from '../call/tag-result.js';

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
 * Build a synthetic `ModelRetryContext` carrying an error attempt for a language
 * model. Used by unit tests that exercise condition predicates without
 * spinning up `generateText`.
 */
export const buildErrorContext = (
  error: unknown,
  model: MockLanguageModel = MockLanguageModel.from(),
): ModelRetryContext<MockLanguageModel> => {
  const attempt: ModelRetryErrorAttempt<MockLanguageModel> = {
    type: 'error',
    error,
    model,
    options: {} as LanguageModelCallOptions,
  };
  return { current: attempt, attempts: [attempt] };
};

/**
 * Build a synthetic `ModelRetryContext` carrying a result attempt for a language
 * model.
 */
export const buildResultContext = (
  result: LanguageModelGenerate = Language.result([]),
  model: MockLanguageModel = MockLanguageModel.from(),
): ModelRetryContext<MockLanguageModel> => {
  const attempt: ModelRetryResultAttempt = {
    type: 'result',
    result,
    finishReason: result.finishReason.unified,
    model,
    options: {} as LanguageModelCallOptions,
  };
  return { current: attempt, attempts: [attempt] };
};

/**
 * Build a synthetic `ModelRetryContext` carrying an error attempt for an image
 * model.
 */
export const buildImageErrorContext = (
  error: unknown,
  model: MockImageModel = MockImageModel.from(),
): ModelRetryContext<MockImageModel> => {
  const attempt: ModelRetryErrorAttempt<MockImageModel> = {
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
export const apiError = (
  options: Partial<ConstructorParameters<typeof APICallError>[0]> = {},
): APICallError =>
  new APICallError({
    message: 'boom',
    url: '',
    requestBodyValues: {},
    statusCode: 500,
    responseHeaders: {},
    responseBody: '',
    data: {},
    ...options,
  });

export const serviceOverloadedError = new APICallError({
  message: `Service overloaded`,
  url: ``,
  requestBodyValues: {},
  statusCode: 529,
  responseHeaders: {},
  responseBody: `{"error": {"message": "Service overloaded"}}`,
  isRetryable: true,
  data: {
    error: {
      message: `Service overloaded`,
    },
  },
});

export const serviceUnavailableError = new APICallError({
  message: `Service unavailable`,
  url: ``,
  requestBodyValues: {},
  statusCode: 503,
  responseHeaders: {},
  responseBody: `{"error": {"message": "Service unavailable"}}`,
  isRetryable: true,
  data: {
    error: {
      message: `Service unavailable`,
    },
  },
});

/** `Error` with `name === 'TimeoutError'`, as `AbortSignal.timeout()` produces. */
export const timeoutError = (): Error => {
  const err = new Error('timed out');
  err.name = 'TimeoutError';
  return err;
};

/** `Error` with `name === 'AbortError'`, as manual `controller.abort()` produces. */
export const abortError = (): Error => {
  const err = new Error('aborted');
  err.name = 'AbortError';
  return err;
};

export const retryableError = Errors.rateLimited();
export const nonRetryableError = Errors.unauthorized();

/** Azure-style content-filter `APICallError` (data code `content_filter`). */
export const contentFilterError = apiError({
  message: 'The response was filtered due to the content management policy.',
  statusCode: 400,
  isRetryable: false,
  data: {
    error: {
      message:
        'The response was filtered due to the content management policy.',
      type: null,
      param: 'prompt',
      code: 'content_filter',
    },
  },
});

export const testUsage = {
  inputTokens: { total: 3, noCache: 0, cacheRead: 0, cacheWrite: 0 },
  outputTokens: { total: 10, text: 0, reasoning: 0 },
};

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

/**
 * Stream parts that finish with `content-filter` before any text deltas are
 * emitted. The retry layer evaluates result-based retryables against a
 * synthetic result built from the finish part when no content has streamed yet.
 */
export const contentFilterStreamChunks: Array<LanguageModelStreamPart> = [
  { type: 'stream-start', warnings: [] },
  {
    type: 'response-metadata',
    id: 'id-0',
    modelId: 'mock-model-id',
    timestamp: new Date(0),
  },
  {
    type: 'finish',
    finishReason: { unified: 'content-filter', raw: undefined },
    usage: testUsage,
    providerMetadata: {
      testProvider: { testKey: 'testValue' },
    },
  },
];

/** A successful embedding result. */
export const mockEmbeddings: EmbeddingModelEmbed = Embedding.result(
  [Embedding.vector(3)],
  { usage: Embedding.usage(5) },
);

/** A successful image generation result. */
export const mockImageResult: ImageModelGenerate = Image.result([Image.png()]);

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

/* ------------------------------------------------------------------ *
 * Call-layer fixtures
 * ------------------------------------------------------------------ */

/**
 * Build a real `generateText` result and present it as its tagged union member.
 *
 * Run through the SDK rather than assembled from a literal, and that is the
 * point: the SDK exposes most of a result through prototype getters, so a
 * hand-built stand-in would not exercise what {@link tagResult} exists to
 * forward. A condition under test sees exactly the object a caller would.
 */
export const callGenerateTextResult = async (
  text: string = mockResultText,
  options?: Parameters<typeof generateText>[0],
): Promise<GenerateTextResultInfo> =>
  tagResult(
    'generateText',
    await generateText({
      model: MockLanguageModel.from(text),
      prompt: 'Hello!',
      ...options,
    } as Parameters<typeof generateText>[0]),
  );

/** A real `embed` result, tagged. */
export const callEmbedResult = async (
  vector: Array<number> = [0.1, 0.2, 0.3],
): Promise<EmbedResultInfo> =>
  tagResult(
    'embed',
    await embed({
      model: MockEmbeddingModel.from([vector]),
      value: 'Hello!',
    }),
  );

/** A real `embedMany` result, tagged. */
export const callEmbedManyResult = async (
  vectors: Array<Array<number>> = [[0.1, 0.2, 0.3]],
): Promise<EmbedManyResultInfo> =>
  tagResult(
    'embedMany',
    await embedMany({
      model: MockEmbeddingModel.from(vectors),
      values: vectors.map((_, i) => `value ${i}`),
    }),
  );

/** A real `generateImage` result, tagged. */
export const callImageResult = async (
  count = 1,
): Promise<GenerateImageResultInfo> =>
  tagResult(
    'generateImage',
    await generateImage({
      model: MockImageModel.from(
        Image.result(Array.from({ length: count }, () => Image.png())),
      ),
      prompt: 'a cat',
    }),
  );

/** A `streamText` result as a contentless stream reports it. */
export const callStreamTextResult = (
  finishReason: CallFinishReason = 'stop',
): StreamTextResultInfo => ({
  operation: 'streamText',
  finishReason,
  usage: {
    inputTokens: 10,
    outputTokens: 0,
    totalTokens: 10,
    inputTokenDetails: {
      noCacheTokens: 10,
      cacheReadTokens: 0,
      cacheWriteTokens: 0,
    },
    outputTokenDetails: { textTokens: 0, reasoningTokens: 0 },
  },
  providerMetadata: undefined,
});

/**
 * Build a synthetic `CallRetryContext` carrying an error attempt for a
 * language model. The call-layer counterpart of {@link buildErrorContext}.
 */
export const buildCallErrorContext = (
  error: unknown,
  model: MockLanguageModel = MockLanguageModel.from(),
): CallRetryContext<MockLanguageModel> => {
  const attempt: CallRetryErrorAttempt<MockLanguageModel> = {
    type: 'error',
    error,
    model,
    options: {} as CallArgs<MockLanguageModel>,
  };
  return { current: attempt, attempts: [attempt] };
};

/**
 * Build a synthetic `CallRetryContext` carrying a result attempt for a
 * language model. The call-layer counterpart of {@link buildResultContext}.
 */
export const buildCallResultContext = (
  result: CallResult<MockLanguageModel>,
  model: MockLanguageModel = MockLanguageModel.from(),
): CallRetryContext<MockLanguageModel> => {
  const attempt: CallRetryResultAttempt<MockLanguageModel> = {
    type: 'result',
    result,
    model,
    options: {} as CallArgs<MockLanguageModel>,
  };
  return { current: attempt, attempts: [attempt] };
};

/** As {@link buildCallResultContext}, for an embedding model. */
export const buildCallEmbeddingResultContext = (
  result: CallResult<MockEmbeddingModel>,
  model: MockEmbeddingModel = MockEmbeddingModel.from(),
): CallRetryContext<MockEmbeddingModel> => {
  const attempt: CallRetryResultAttempt<MockEmbeddingModel> = {
    type: 'result',
    result,
    model,
    options: {} as CallArgs<MockEmbeddingModel>,
  };
  return { current: attempt, attempts: [attempt] };
};

/** As {@link buildCallResultContext}, for an image model. */
export const buildCallImageResultContext = (
  result: CallResult<MockImageModel>,
  model: MockImageModel = MockImageModel.from(),
): CallRetryContext<MockImageModel> => {
  const attempt: CallRetryResultAttempt<MockImageModel> = {
    type: 'result',
    result,
    model,
    options: {} as CallArgs<MockImageModel>,
  };
  return { current: attempt, attempts: [attempt] };
};

/** As {@link buildCallErrorContext}, for an image model. */
export const buildCallImageErrorContext = (
  error: unknown,
  model: MockImageModel = MockImageModel.from(),
): CallRetryContext<MockImageModel> => {
  const attempt: CallRetryErrorAttempt<MockImageModel> = {
    type: 'error',
    error,
    model,
    options: {} as CallArgs<MockImageModel>,
  };
  return { current: attempt, attempts: [attempt] };
};
