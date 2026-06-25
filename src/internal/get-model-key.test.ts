import { describe, expect, it } from 'vitest';
import { getModelKey } from './get-model-key.js';
import {
  MockEmbeddingModel,
  MockImageModel,
  MockLanguageModel,
} from './test-utils.js';

describe('getModelKey', () => {
  it(`should return provider/modelId for a language model`, () => {
    // Arrange
    const model = MockLanguageModel.from();

    // Act
    const key = getModelKey(model);

    // Assert
    expect(key).toBe(`${model.provider}/${model.modelId}`);
  });

  it(`should return provider/modelId for an embedding model`, () => {
    // Arrange
    const model = MockEmbeddingModel.from();

    // Act
    const key = getModelKey(model);

    // Assert
    expect(key).toBe(`${model.provider}/${model.modelId}`);
  });

  it(`should return provider/modelId for an image model`, () => {
    // Arrange
    const model = MockImageModel.from();

    // Act
    const key = getModelKey(model);

    // Assert
    expect(key).toBe(`${model.provider}/${model.modelId}`);
  });
});
