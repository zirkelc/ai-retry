import { generateText, Output } from 'ai';
import { describe, expect, it } from 'vitest';
import { z } from 'zod';
import {
  createRetryableModel,
  invalidJson,
  MockLanguageModel,
  notJson,
  personSchema,
  validJson,
} from '../test-utils.js';
import { createResultAPI } from './result.js';

const { schemaInvalid } = createResultAPI<MockLanguageModel>();

describe('schemaInvalid', () => {
  it(`should not switch when JSON matches schema`, async () => {
    // Arrange
    const baseModel = MockLanguageModel.from(validJson);
    const fallback = MockLanguageModel.from(validJson);

    // Act
    const result = await generateText({
      model: createRetryableModel({
        model: baseModel,
        retries: [schemaInvalid().switch({ model: fallback })],
      }),
      prompt: 'Generate a person',
      maxRetries: 0,
      output: Output.object({ schema: personSchema }),
    });

    // Assert
    expect(baseModel.doGenerate).toHaveBeenCalledTimes(1);
    expect(fallback.doGenerate).toHaveBeenCalledTimes(0);
    expect(result.output).toEqual({ name: 'Alice', age: 30 });
  });

  it(`should switch when JSON does not match schema`, async () => {
    // Arrange
    const baseModel = MockLanguageModel.from(invalidJson);
    const fallback = MockLanguageModel.from(validJson);

    // Act
    const result = await generateText({
      model: createRetryableModel({
        model: baseModel,
        retries: [schemaInvalid().switch({ model: fallback })],
      }),
      prompt: 'Generate a person',
      maxRetries: 0,
      output: Output.object({ schema: personSchema }),
    });

    // Assert
    expect(baseModel.doGenerate).toHaveBeenCalledTimes(1);
    expect(fallback.doGenerate).toHaveBeenCalledTimes(1);
    expect(result.output).toEqual({ name: 'Alice', age: 30 });
  });

  it(`should switch when response is not valid JSON`, async () => {
    // Arrange
    const baseModel = MockLanguageModel.from(notJson);
    const fallback = MockLanguageModel.from(validJson);

    // Act
    const result = await generateText({
      model: createRetryableModel({
        model: baseModel,
        retries: [schemaInvalid().switch({ model: fallback })],
      }),
      prompt: 'Generate a person',
      maxRetries: 0,
      output: Output.object({ schema: personSchema }),
    });

    // Assert
    expect(baseModel.doGenerate).toHaveBeenCalledTimes(1);
    expect(fallback.doGenerate).toHaveBeenCalledTimes(1);
    expect(result.output).toEqual({ name: 'Alice', age: 30 });
  });

  it(`should not switch when no schema is provided`, async () => {
    // Arrange
    const baseModel = MockLanguageModel.from(notJson);
    const fallback = MockLanguageModel.from(validJson);

    // Act
    const result = await generateText({
      model: createRetryableModel({
        model: baseModel,
        retries: [schemaInvalid().switch({ model: fallback })],
      }),
      prompt: 'just say hi',
      maxRetries: 0,
    });

    // Assert
    expect(baseModel.doGenerate).toHaveBeenCalledTimes(1);
    expect(fallback.doGenerate).toHaveBeenCalledTimes(0);
    expect(result.text).toBe(notJson);
  });
});
