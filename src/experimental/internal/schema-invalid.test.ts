import { generateText, Output } from 'ai';
import { describe, expect, it } from 'vitest';
import { z } from 'zod';
import { createRetryable } from '../../create-retryable-model.js';
import {
  generateTextResult,
  MockLanguageModel,
} from '../../internal/test-utils.js';
import { createResultAPI } from './result.js';

const { schemaInvalid } = createResultAPI<MockLanguageModel>();

const personSchema = z.object({
  name: z.string(),
  age: z.number(),
});

const validJson = JSON.stringify({ name: 'Alice', age: 30 });
const invalidJson = JSON.stringify({ name: 123 });
const notJson = 'this is not json';

describe('schemaInvalid', () => {
  it(`should not switch when JSON matches schema`, async () => {
    // Arrange
    const baseModel = new MockLanguageModel({
      doGenerate: generateTextResult(validJson),
    });
    const fallback = new MockLanguageModel({
      doGenerate: generateTextResult(validJson),
    });

    // Act
    const result = await generateText({
      model: createRetryable({
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
    const baseModel = new MockLanguageModel({
      doGenerate: generateTextResult(invalidJson),
    });
    const fallback = new MockLanguageModel({
      doGenerate: generateTextResult(validJson),
    });

    // Act
    const result = await generateText({
      model: createRetryable({
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
    const baseModel = new MockLanguageModel({
      doGenerate: generateTextResult(notJson),
    });
    const fallback = new MockLanguageModel({
      doGenerate: generateTextResult(validJson),
    });

    // Act
    const result = await generateText({
      model: createRetryable({
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
    const baseModel = new MockLanguageModel({
      doGenerate: generateTextResult(notJson),
    });
    const fallback = new MockLanguageModel({
      doGenerate: generateTextResult(validJson),
    });

    // Act
    const result = await generateText({
      model: createRetryable({
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
