import { openai } from '@ai-sdk/openai';
import { describe, expect, expectTypeOf, it } from 'vitest';
import { resolveModel } from './resolve-model.js';
import type { EmbeddingModel, LanguageModel } from './types.js';

describe('resolveModel', () => {
  it('should resolve a model string to a model instance', () => {
    const model = resolveModel('openai/gpt-4.1');
    expectTypeOf(model).toEqualTypeOf<LanguageModel>();
  });

  it('should return a language model instance', () => {
    const openaiModel = openai('gpt-4.1');
    const model = resolveModel(openaiModel);

    expectTypeOf(model).toEqualTypeOf<LanguageModel>();
  });

  it('should return an embedding model instance', () => {
    const openaiModel = openai.embedding('text-embedding-3-large');
    const model = resolveModel(openaiModel);

    expectTypeOf(model).toEqualTypeOf<EmbeddingModel>();
  });
});
