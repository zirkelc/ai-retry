import type { LanguageModelV2 } from '@ai-sdk/provider';
import type { generateText, streamText, TextStreamPart } from 'ai';
import { MockLanguageModelV2 } from 'ai/test';

type StreamText = Parameters<typeof streamText>[0];
type GenerateText = Parameters<typeof generateText>[0];

export type LanguageModelV2GenerateFn = LanguageModelV2['doGenerate'];
export type LanguageModelV2StreamFn = LanguageModelV2['doStream'];

export type LanguageModelV2GenerateResult = Awaited<
  ReturnType<LanguageModelV2GenerateFn>
>;

export type LanguageModelV2StreamResult = Awaited<
  ReturnType<LanguageModelV2StreamFn>
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
  return currentMockModelId();
};

const provider = 'mock-provider';
export const currentMockModelId = () => `mock-model-${mockModelCounter}`;

export const createMockModel = (
  resultOrFunction:
    | LanguageModelV2GenerateResult
    | LanguageModelV2GenerateFn
    | Error,
) => {
  const modelId = generateMockModelId();

  if (resultOrFunction instanceof Error)
    return new MockLanguageModelV2({
      provider,
      modelId,
      doGenerate: async () => {
        throw resultOrFunction;
      },
    });

  if (typeof resultOrFunction === 'function')
    return new MockLanguageModelV2({
      provider,
      modelId,
      doGenerate: resultOrFunction,
    });

  return new MockLanguageModelV2({
    provider,
    modelId,
    doGenerate: async () => resultOrFunction,
  });
};

export const createMockStreamingModel = (
  resultOrFunction:
    | LanguageModelV2StreamResult
    | LanguageModelV2StreamFn
    | Error,
) => {
  const modelId = generateMockModelId();

  if (resultOrFunction instanceof Error)
    return new MockLanguageModelV2({
      provider,
      modelId,
      doStream: async () => {
        throw resultOrFunction;
      },
    });

  if (typeof resultOrFunction === 'function')
    return new MockLanguageModelV2({
      provider,
      modelId,
      doStream: resultOrFunction,
    });

  return new MockLanguageModelV2({
    provider,
    modelId,
    doStream: async () => resultOrFunction,
  });
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
