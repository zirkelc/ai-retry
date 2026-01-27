import type { generateText, streamText, TextStreamPart } from 'ai';
import { vi } from 'vitest';
import type {
  EmbeddingModel,
  EmbeddingModelEmbed,
  LanguageModel,
  LanguageModelGenerate,
  LanguageModelStream,
} from './types.js';

type StreamText = Parameters<typeof streamText>[0];
type GenerateText = Parameters<typeof generateText>[0];

export type LanguageModelGenerateFn = LanguageModel['doGenerate'];
export type LanguageModelStreamFn = LanguageModel['doStream'];
export type EmbeddingModelEmbedFn = EmbeddingModel['doEmbed'];

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
  readonly supportsParallelCalls: EmbeddingModel['supportsParallelCalls'] =
    true;

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

export const errorFromChunks = (
  chunks: TextStreamPart<any>[],
): unknown | null => {
  const errorChunk = chunks.find((chunk) => chunk.type === 'error');
  return errorChunk && errorChunk.type === 'error' ? errorChunk.error : null;
};
