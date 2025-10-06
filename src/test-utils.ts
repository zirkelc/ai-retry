import type { LanguageModelV2 } from '@ai-sdk/provider';
import type { generateText, streamText, TextStreamPart } from 'ai';
import { vi } from 'vitest';
import type { EmbeddingModelV2 } from './types.js';

type StreamText = Parameters<typeof streamText>[0];
type GenerateText = Parameters<typeof generateText>[0];

export type LanguageModelV2GenerateFn = LanguageModelV2['doGenerate'];
export type LanguageModelV2StreamFn = LanguageModelV2['doStream'];

export type LanguageModelV2Generate = Awaited<
  ReturnType<LanguageModelV2GenerateFn>
>;
export type LanguageModelV2Stream = Awaited<
  ReturnType<LanguageModelV2StreamFn>
>;

export type EmbeddingModelV2EmbedFn = EmbeddingModelV2<number>['doEmbed'];
export type EmbeddingModelV2Embed = Awaited<
  ReturnType<EmbeddingModelV2EmbedFn>
>;

const mockGenerateId = () => 'aitxt-mock-id';
const mockCurrentDate = () => new Date(0);

export const mockGenerateOptions: Partial<GenerateText> = {
  _internal: {
    generateId: mockGenerateId,
    currentDate: mockCurrentDate,
  },
};

export const mockStreamOptions: Pick<StreamText, '_internal'> = {
  _internal: {
    generateId: mockGenerateId,
    currentDate: mockCurrentDate,
  },
};

let mockModelCounter = 0;
const generateMockModelId = () => {
  mockModelCounter += 1;
  return `mock-model-${mockModelCounter}`;
};

export class MockLanguageModel implements LanguageModelV2 {
  readonly specificationVersion = 'v2';

  readonly supportedUrls: LanguageModelV2['supportedUrls'];
  readonly provider: LanguageModelV2['provider'];
  readonly modelId: LanguageModelV2['modelId'];

  doGenerate: LanguageModelV2['doGenerate'];
  doStream: LanguageModelV2['doStream'];

  constructor({
    doGenerate = (): never => {
      throw new Error('Not implemented');
    },
    doStream = (): never => {
      throw new Error('Not implemented');
    },
  }: {
    doGenerate?: LanguageModelV2Generate | LanguageModelV2GenerateFn | Error;
    doStream?: LanguageModelV2Stream | LanguageModelV2StreamFn | Error;
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

export class MockEmbeddingModel implements EmbeddingModelV2<number> {
  readonly specificationVersion = 'v2';

  readonly supportedUrls: LanguageModelV2['supportedUrls'];
  readonly provider: LanguageModelV2['provider'];
  readonly modelId: LanguageModelV2['modelId'];
  readonly maxEmbeddingsPerCall = 1;
  readonly supportsParallelCalls = true;

  doEmbed: EmbeddingModelV2['doEmbed'];

  constructor({
    doEmbed = (): never => {
      throw new Error('Not implemented');
    },
  }: {
    doEmbed?: EmbeddingModelV2Embed | EmbeddingModelV2EmbedFn | Error;
  } = {}) {
    this.provider = 'mock-provider';
    this.modelId = generateMockModelId();
    this.supportedUrls = {};
    this.doEmbed = vi.fn(async (opts) => {
      if (doEmbed instanceof Error) throw doEmbed;
      if (typeof doEmbed === 'function') return doEmbed(opts);
      return doEmbed;
    });
  }
}

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
