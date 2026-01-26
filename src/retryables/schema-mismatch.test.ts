import { generateText, Output } from 'ai';
import { describe, expect, it } from 'vitest';
import { z } from 'zod';
import { createRetryable } from '../create-retryable-model.js';
import {
  generateEmptyResult,
  generateTextResult,
  MockLanguageModel,
} from '../test-utils.js';
import { schemaMismatch } from './schema-mismatch.js';

const validJson = JSON.stringify({ name: `Alice`, age: 30 });
const invalidJson = JSON.stringify({ name: 123 });
const notJson = `this is not json`;

const personSchema = z.object({
  name: z.string(),
  age: z.number(),
});

describe(`schemaMismatch`, () => {
  it(`should not retry when JSON matches schema`, async () => {
    // Arrange
    const baseModel = new MockLanguageModel({
      doGenerate: generateTextResult(validJson),
    });
    const retryModel = new MockLanguageModel({
      doGenerate: generateTextResult(validJson),
    });

    // Act
    const result = await generateText({
      model: createRetryable({
        model: baseModel,
        retries: [schemaMismatch(retryModel)],
      }),
      prompt: `Generate a person`,
      maxRetries: 0,
      output: Output.object({ schema: personSchema }),
    });

    // Assert
    expect(baseModel.doGenerate).toHaveBeenCalledTimes(1);
    expect(retryModel.doGenerate).toHaveBeenCalledTimes(0);
    expect(result.output).toEqual({ name: `Alice`, age: 30 });
  });

  it(`should retry when JSON does not match schema`, async () => {
    // Arrange
    const baseModel = new MockLanguageModel({
      doGenerate: generateTextResult(invalidJson),
    });
    const retryModel = new MockLanguageModel({
      doGenerate: generateTextResult(validJson),
    });

    // Act
    const result = await generateText({
      model: createRetryable({
        model: baseModel,
        retries: [schemaMismatch(retryModel)],
      }),
      prompt: `Generate a person`,
      maxRetries: 0,
      output: Output.object({ schema: personSchema }),
    });

    // Assert
    expect(baseModel.doGenerate).toHaveBeenCalledTimes(1);
    expect(retryModel.doGenerate).toHaveBeenCalledTimes(1);
    expect(result.output).toEqual({ name: `Alice`, age: 30 });
  });

  it(`should retry when response is not valid JSON`, async () => {
    // Arrange
    const baseModel = new MockLanguageModel({
      doGenerate: generateTextResult(notJson),
    });
    const retryModel = new MockLanguageModel({
      doGenerate: generateTextResult(validJson),
    });

    // Act
    const result = await generateText({
      model: createRetryable({
        model: baseModel,
        retries: [schemaMismatch(retryModel)],
      }),
      prompt: `Generate a person`,
      maxRetries: 0,
      output: Output.object({ schema: personSchema }),
    });

    // Assert
    expect(baseModel.doGenerate).toHaveBeenCalledTimes(1);
    expect(retryModel.doGenerate).toHaveBeenCalledTimes(1);
    expect(result.output).toEqual({ name: `Alice`, age: 30 });
  });

  it(`should not retry when no responseFormat schema is present`, async () => {
    // Arrange
    const baseModel = new MockLanguageModel({
      doGenerate: generateTextResult(invalidJson),
    });
    const retryModel = new MockLanguageModel({
      doGenerate: generateTextResult(validJson),
    });

    // Act
    const result = await generateText({
      model: createRetryable({
        model: baseModel,
        retries: [schemaMismatch(retryModel)],
      }),
      prompt: `Generate something`,
      maxRetries: 0,
    });

    // Assert
    expect(baseModel.doGenerate).toHaveBeenCalledTimes(1);
    expect(retryModel.doGenerate).toHaveBeenCalledTimes(0);
    expect(result.text).toBe(invalidJson);
  });

  it(`should not retry on error attempt`, async () => {
    // Arrange
    const error = new Error(`model failed`);
    const baseModel = new MockLanguageModel({
      doGenerate: error,
    });
    const retryModel = new MockLanguageModel({
      doGenerate: generateTextResult(validJson),
    });

    // Act
    const result = generateText({
      model: createRetryable({
        model: baseModel,
        retries: [schemaMismatch(retryModel)],
      }),
      prompt: `Generate a person`,
      maxRetries: 0,
      output: Output.object({ schema: personSchema }),
    });

    // Assert
    await expect(result).rejects.toThrow();
    expect(baseModel.doGenerate).toHaveBeenCalledTimes(1);
    expect(retryModel.doGenerate).toHaveBeenCalledTimes(0);
  });

  it(`should retry with Output.array() when elements don't match schema`, async () => {
    // Arrange
    const validArrayJson = JSON.stringify({
      elements: [{ name: `Alice`, age: 30 }],
    });
    const invalidArrayJson = JSON.stringify({ elements: [{ name: 123 }] });
    const baseModel = new MockLanguageModel({
      doGenerate: generateTextResult(invalidArrayJson),
    });
    const retryModel = new MockLanguageModel({
      doGenerate: generateTextResult(validArrayJson),
    });

    // Act
    const result = await generateText({
      model: createRetryable({
        model: baseModel,
        retries: [schemaMismatch(retryModel)],
      }),
      prompt: `Generate people`,
      maxRetries: 0,
      output: Output.array({ element: personSchema }),
    });

    // Assert
    expect(baseModel.doGenerate).toHaveBeenCalledTimes(1);
    expect(retryModel.doGenerate).toHaveBeenCalledTimes(1);
    expect(result.output).toEqual([{ name: `Alice`, age: 30 }]);
  });

  it(`should retry with Output.choice() when choice is not in options`, async () => {
    // Arrange
    const validChoiceJson = JSON.stringify({ result: `yes` });
    const invalidChoiceJson = JSON.stringify({ result: `maybe` });
    const baseModel = new MockLanguageModel({
      doGenerate: generateTextResult(invalidChoiceJson),
    });
    const retryModel = new MockLanguageModel({
      doGenerate: generateTextResult(validChoiceJson),
    });

    // Act
    const result = await generateText({
      model: createRetryable({
        model: baseModel,
        retries: [schemaMismatch(retryModel)],
      }),
      prompt: `Yes or no?`,
      maxRetries: 0,
      output: Output.choice({ options: [`yes`, `no`] }),
    });

    // Assert
    expect(baseModel.doGenerate).toHaveBeenCalledTimes(1);
    expect(retryModel.doGenerate).toHaveBeenCalledTimes(1);
    expect(result.output).toBe(`yes`);
  });

  it(`should not retry when no text content`, async () => {
    // Arrange
    const baseModel = new MockLanguageModel({
      doGenerate: generateEmptyResult,
    });
    const retryModel = new MockLanguageModel({
      doGenerate: generateTextResult(validJson),
    });

    // Act
    const result = generateText({
      model: createRetryable({
        model: baseModel,
        retries: [schemaMismatch(retryModel)],
      }),
      prompt: `Generate a person`,
      maxRetries: 0,
      output: Output.object({ schema: personSchema }),
    });

    // Assert — retryable skips (no text), but Output.object() throws on empty text
    await expect(result).rejects.toThrow();
    expect(baseModel.doGenerate).toHaveBeenCalledTimes(1);
    expect(retryModel.doGenerate).toHaveBeenCalledTimes(0);
  });
});
