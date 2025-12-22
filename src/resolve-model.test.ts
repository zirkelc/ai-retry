import {} from '@ai-sdk/gateway';
import { openai } from '@ai-sdk/openai';
import { describe, expect, it } from 'vitest';
import { parseRetryHeaders } from './parse-retry-headers.js';
import { resolveModel } from './resolve-model.js';

describe('resolveModel', () => {
  it('should resolve a model string to a model instance', () => {
    const model = resolveModel('openai/gpt-4.1');
    expect(model.provider).toBe('gateway');
    expect(model.modelId).toBe('openai/gpt-4.1');
  });

  it('should return a language model instance', () => {
    const openaiModel = openai('gpt-4.1');
    const model = resolveModel(openaiModel);

    expect(model).toBe(openaiModel);
    expect(model.provider).toBe(openaiModel.provider);
    expect(model.modelId).toBe(openaiModel.modelId);
  });

  it('should return an embedding model instance', () => {
    const openaiModel = openai.embedding('text-embedding-3-large');
    const model = resolveModel(openaiModel);

    expect(model).toBe(openaiModel);
    expect(model.provider).toBe(openaiModel.provider);
    expect(model.modelId).toBe(openaiModel.modelId);
  });
});
