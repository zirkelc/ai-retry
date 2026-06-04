import { convertArrayToReadableStream } from '@ai-sdk/provider-utils/test';
import type { Tracer } from '@opentelemetry/api';
import {
  BasicTracerProvider,
  InMemorySpanExporter,
  type ReadableSpan,
  SimpleSpanProcessor,
} from '@opentelemetry/sdk-trace-base';
import {
  APICallError,
  type generateText,
  type streamText,
  type TextStreamPart,
} from 'ai';
import { vi } from 'vitest';
import type {
  EmbeddingModel,
  EmbeddingModelCallOptions,
  EmbeddingModelEmbed,
  ImageModel,
  ImageModelCallOptions,
  ImageModelGenerate,
  LanguageModel,
  LanguageModelCallOptions,
  LanguageModelGenerate,
  LanguageModelResult,
  LanguageModelStream,
  LanguageModelStreamPart,
  RetryContext,
  RetryErrorAttempt,
  RetryResultAttempt,
} from '../types.js';

/**
 * Re-exported for test convenience: the auto-detecting factory builds a
 * retryable model for any family from a single import.
 */
export { createRetryable } from './create-retryable.js';

type StreamText = Parameters<typeof streamText>[0];
type GenerateText = Parameters<typeof generateText>[0];

export type LanguageModelGenerateFn = LanguageModel['doGenerate'];
export type LanguageModelStreamFn = LanguageModel['doStream'];
export type EmbeddingModelEmbedFn = EmbeddingModel['doEmbed'];
export type ImageModelGenerateFn = ImageModel['doGenerate'];

const mockGenerateId = () => 'aitxt-mock-id';

export const mockGenerateOptions: Partial<GenerateText> = {
  _internal: {
    generateId: mockGenerateId,
  },
};

/**
 * Note: `_internal.now` controls most timestamps in `streamText`, but the AI SDK has a bug where
 * the `finish-step` response timestamp uses `new Date()` directly instead of `new Date(now())` in
 * the error streaming path. Tests that hit that path need `vi.useFakeTimers()` as a workaround.
 * @see https://github.com/vercel/ai/blob/main/packages/ai/src/generate-text/stream-text.ts
 */
export const mockStreamOptions: Pick<StreamText, '_internal'> = {
  _internal: {
    generateId: mockGenerateId,
    now: () => 0,
  },
};

let mockModelCounter = 0;
const generateMockModelId = () => {
  mockModelCounter += 1;
  return `mock-model-${mockModelCounter}`;
};

export class MockLanguageModel implements LanguageModel {
  readonly specificationVersion = 'v3';

  readonly supportedUrls: LanguageModel['supportedUrls'];
  readonly provider: LanguageModel['provider'];
  readonly modelId: LanguageModel['modelId'];

  doGenerate: LanguageModel['doGenerate'];
  doStream: LanguageModel['doStream'];

  constructor({
    doGenerate = (): never => {
      throw new Error('Not implemented');
    },
    doStream = (): never => {
      throw new Error('Not implemented');
    },
  }: {
    doGenerate?: LanguageModelGenerate | LanguageModelGenerateFn | Error;
    doStream?: LanguageModelStream | LanguageModelStreamFn | Error;
  } = {}) {
    this.provider = 'mock-provider';
    this.modelId = generateMockModelId();
    this.supportedUrls = {};
    this.doGenerate = vi.fn(async (opts) => {
      if (doGenerate instanceof Error) throw doGenerate;
      if (typeof doGenerate === 'function') return doGenerate(opts);
      return doGenerate;
    });
    this.doStream = vi.fn(async (opts) => {
      if (doStream instanceof Error) throw doStream;
      if (typeof doStream === 'function') return doStream(opts);
      return doStream;
    });
  }
}

export class MockEmbeddingModel implements EmbeddingModel {
  readonly specificationVersion = 'v3';

  readonly provider: EmbeddingModel['provider'];
  readonly modelId: EmbeddingModel['modelId'];
  readonly maxEmbeddingsPerCall: EmbeddingModel['maxEmbeddingsPerCall'] = 1;
  readonly supportsParallelCalls: EmbeddingModel['supportsParallelCalls'] = true;

  doEmbed: EmbeddingModel['doEmbed'];

  constructor({
    doEmbed = (): never => {
      throw new Error('Not implemented');
    },
  }: {
    doEmbed?: EmbeddingModelEmbed | EmbeddingModelEmbedFn | Error;
  } = {}) {
    this.provider = 'mock-provider';
    this.modelId = generateMockModelId();
    this.doEmbed = vi.fn(async (opts) => {
      if (doEmbed instanceof Error) throw doEmbed;
      if (typeof doEmbed === 'function') return doEmbed(opts);
      return doEmbed;
    });
  }
}

export class MockImageModel implements ImageModel {
  readonly specificationVersion = 'v3';

  readonly provider: ImageModel['provider'];
  readonly modelId: ImageModel['modelId'];
  readonly maxImagesPerCall: ImageModel['maxImagesPerCall'] = 1;

  doGenerate: ImageModel['doGenerate'];

  constructor({
    doGenerate = (): never => {
      throw new Error(`Not implemented`);
    },
  }: {
    doGenerate?: ImageModelGenerate | ImageModelGenerateFn | Error;
  } = {}) {
    this.provider = `mock-provider`;
    this.modelId = generateMockModelId();
    this.doGenerate = vi.fn(async (opts) => {
      if (doGenerate instanceof Error) throw doGenerate;
      if (typeof doGenerate === `function`) return doGenerate(opts);
      return doGenerate;
    });
  }
}

export const generateTextResult = (text: string): LanguageModelGenerate => ({
  finishReason: { unified: 'stop', raw: undefined },
  usage: {
    inputTokens: { total: 10, noCache: 0, cacheRead: 0, cacheWrite: 0 },
    outputTokens: { total: 20, text: 0, reasoning: 0 },
  },
  content: [{ type: 'text', text }],
  warnings: [],
});

export const generateEmptyResult: LanguageModelGenerate = {
  finishReason: { unified: 'stop', raw: undefined },
  usage: {
    inputTokens: { total: 10, noCache: 0, cacheRead: 0, cacheWrite: 0 },
    outputTokens: { total: 20, text: 0, reasoning: 0 },
  },
  content: [],
  warnings: [],
};

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
  model: MockLanguageModel = new MockLanguageModel(),
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
  result: LanguageModelGenerate = generateEmptyResult,
  model: MockLanguageModel = new MockLanguageModel(),
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
  model: MockImageModel = new MockImageModel(),
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
 * Build an `APICallError` with sensible defaults. Override any field.
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

/** Retryable API error: HTTP 429 rate limit. */
export const retryableError = apiError({
  message: 'Rate limit exceeded',
  statusCode: 429,
  isRetryable: true,
  responseBody:
    '{"error": {"message": "Rate limit exceeded", "code": "rate_limit_exceeded"}}',
  data: {
    error: { message: 'Rate limit exceeded', code: 'rate_limit_exceeded' },
  },
});

/** Non-retryable API error: HTTP 401 invalid key. */
export const nonRetryableError = apiError({
  message: 'Invalid API key',
  statusCode: 401,
  isRetryable: false,
  responseBody:
    '{"error": {"message": "Invalid API key", "code": "invalid_api_key"}}',
  data: {
    error: { message: 'Invalid API key', code: 'invalid_api_key' },
  },
});

export const testUsage = {
  inputTokens: { total: 3, noCache: 0, cacheRead: 0, cacheWrite: 0 },
  outputTokens: { total: 10, text: 0, reasoning: 0 },
};

export const mockResultText = 'Hello, world!';

/** A successful generate result with text content. */
export const mockResult: LanguageModelResult =
  generateTextResult(mockResultText);

/** A generate result that finished due to content filtering. */
export const contentFilterResult: LanguageModelResult = {
  ...generateEmptyResult,
  finishReason: { unified: 'content-filter', raw: undefined },
};

/** Stream parts for a full successful stream, including response metadata. */
export const mockStreamChunks: Array<LanguageModelStreamPart> = [
  { type: 'stream-start', warnings: [] },
  {
    type: 'response-metadata',
    id: 'id-0',
    modelId: 'mock-model-id',
    timestamp: new Date(0),
  },
  { type: 'text-start', id: '1' },
  { type: 'text-delta', id: '1', delta: 'Hello' },
  { type: 'text-delta', id: '1', delta: ', ' },
  { type: 'text-delta', id: '1', delta: 'world!' },
  { type: 'text-end', id: '1' },
  {
    type: 'finish',
    finishReason: { unified: 'stop', raw: undefined },
    usage: {
      inputTokens: { total: 10, noCache: 0, cacheRead: 0, cacheWrite: 0 },
      outputTokens: { total: 20, text: 0, reasoning: 0 },
    },
  },
];

/** A successful embedding result. */
export const mockEmbeddings: EmbeddingModelEmbed = {
  embeddings: [[0.1, 0.2, 0.3]],
  usage: { tokens: 5 },
  warnings: [],
};

/** Valid base64 PNG image (1x1 transparent pixel). */
export const validBase64Image = `iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg==`;

/** A successful image generation result. */
export const mockImageResult: ImageModelGenerate = {
  images: [validBase64Image],
  warnings: [],
  response: {
    timestamp: new Date(),
    modelId: 'mock-model',
    headers: undefined,
  },
};

/** Stream parts for a successful stream: content then a `stop` finish. */
export const successStreamChunks = (
  text: string,
): Array<LanguageModelStreamPart> => [
  { type: 'stream-start', warnings: [] },
  { type: 'text-start', id: '1' },
  { type: 'text-delta', id: '1', delta: text },
  { type: 'text-end', id: '1' },
  {
    type: 'finish',
    finishReason: { unified: 'stop', raw: 'stop' },
    usage: testUsage,
  },
];

/** Stream parts that error before any content is emitted. */
export const errorStreamChunks = (
  error: unknown,
): Array<LanguageModelStreamPart> => [
  { type: 'stream-start', warnings: [] },
  { type: 'error', error },
];

/** Wrap stream parts into a `doStream` result. */
export const mockStream = (
  chunks: Array<LanguageModelStreamPart>,
): LanguageModelStream => ({
  stream: convertArrayToReadableStream(chunks),
});

/** Read a stream to completion to drive a consumer's processing logic. */
export const drainStream = async (
  stream: ReadableStream<LanguageModelStreamPart>,
): Promise<void> => {
  const reader = stream.getReader();
  while (true) {
    const { done } = await reader.read();
    if (done) break;
  }
};

/** Read a stream to completion, collecting all parts. */
export const collectParts = async (
  stream: ReadableStream<LanguageModelStreamPart>,
): Promise<Array<LanguageModelStreamPart>> => {
  const parts: Array<LanguageModelStreamPart> = [];
  const reader = stream.getReader();
  while (true) {
    const { done, value } = await reader.read();
    if (done) break;
    parts.push(value);
  }
  return parts;
};

/** Concatenate the text deltas (raw parts carry `delta`, not `text`). */
export const partsToText = (parts: Array<LanguageModelStreamPart>): string =>
  parts.map((p) => (p.type === 'text-delta' ? p.delta : '')).join('');

export const languageCallOptions = {
  prompt: [{ role: 'user', content: [{ type: 'text', text: 'Hello!' }] }],
} as LanguageModelCallOptions;

export const embeddingCallOptions = {
  values: ['Hello!'],
} as EmbeddingModelCallOptions;

export const imageCallOptions = {
  prompt: 'A sunset over mountains',
  n: 1,
} as ImageModelCallOptions;

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
