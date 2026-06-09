import { generateText, Output } from 'ai';
import { describe, expect, it } from 'vitest';
import { z } from 'zod';
import {
  createRetryableModel,
  generateTextResult,
  MockLanguageModel,
} from '../../internal/test-utils.js';
import { schemaInvalid } from './index.js';

const validJson = JSON.stringify({ name: 'Alice', age: 30 });
const invalidJson = JSON.stringify({ name: 123 });

const personSchema = z.object({
  name: z.string(),
  age: z.number(),
});

describe('schemaInvalid', () => {
  describe('generateText', () => {
    it('should not switch when JSON matches the schema', async () => {
      // Arrange
      const baseModel = new MockLanguageModel({
        doGenerate: generateTextResult(validJson),
      });
      const retryModel = new MockLanguageModel({
        doGenerate: generateTextResult(validJson),
      });

      // Act
      const out = await generateText({
        model: createRetryableModel({
          model: baseModel,
          retries: [schemaInvalid().switch({ model: retryModel })],
        }),
        prompt: 'Generate a person',
        maxRetries: 0,
        output: Output.object({ schema: personSchema }),
      });

      // Assert
      expect(baseModel.doGenerate).toHaveBeenCalledTimes(1);
      expect(retryModel.doGenerate).toHaveBeenCalledTimes(0);
      expect(out.output).toEqual({ name: 'Alice', age: 30 });
    });

    it('should switch when JSON does not match the schema', async () => {
      // Arrange
      const baseModel = new MockLanguageModel({
        doGenerate: generateTextResult(invalidJson),
      });
      const retryModel = new MockLanguageModel({
        doGenerate: generateTextResult(validJson),
      });

      // Act
      const out = await generateText({
        model: createRetryableModel({
          model: baseModel,
          retries: [schemaInvalid().switch({ model: retryModel })],
        }),
        prompt: 'Generate a person',
        maxRetries: 0,
        output: Output.object({ schema: personSchema }),
      });

      // Assert
      expect(baseModel.doGenerate).toHaveBeenCalledTimes(1);
      expect(retryModel.doGenerate).toHaveBeenCalledTimes(1);
      expect(out.output).toEqual({ name: 'Alice', age: 30 });
    });
  });
});
