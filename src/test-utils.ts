import type { AISDKError, LanguageModelV2 } from '@ai-sdk/provider';
import { MockLanguageModelV2 } from 'ai/test';

type LanguageModelV2GenerateFn = LanguageModelV2['doGenerate'];

type LanguageModelV2GenerateResult = Awaited<
  ReturnType<LanguageModelV2GenerateFn>
>;

let mockModelCounter = 0;
const generateMockModelId = () => {
  mockModelCounter += 1;
  return currentMockModelId();
};

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
      provider: 'mock-provider',
      modelId,
      doGenerate: async () => {
        throw resultOrFunction;
      },
    });

  if (typeof resultOrFunction === 'function')
    return new MockLanguageModelV2({
      provider: 'mock-provider',
      modelId,

      doGenerate: resultOrFunction,
    });

  return new MockLanguageModelV2({
    provider: 'mock-provider',
    modelId,
    doGenerate: async () => resultOrFunction,
  });
};
