import { streamText, tool } from 'ai';
import { describe, expectTypeOf, it } from 'vitest';
import { z } from 'zod';
import { MockLanguageModel } from '../../internal/test-utils.js';
import { createRetryableStreamText } from './create-retryable-stream-text.js';

describe('createRetryableStreamText', () => {
  it('should thread tool typing from the call args through to the result', async () => {
    // Arrange
    const tools = {
      weather: tool({
        description: 'Get the weather',
        inputSchema: z.object({ city: z.string() }),
        execute: async ({ city }) => `sunny in ${city}`,
      }),
    };
    const retryableStreamText = createRetryableStreamText({
      model: new MockLanguageModel(),
      retries: [],
    });

    // Act
    const result = await retryableStreamText({ prompt: 'hi', tools });
    const direct = streamText({
      model: new MockLanguageModel(),
      prompt: 'hi',
      tools,
    });

    // Assert — the adapter's result is the same fully-typed StreamTextResult
    // a direct streamText call with the same tools would produce.
    expectTypeOf(result).toEqualTypeOf<typeof direct>();
  });

  it('should default the tool set when no tools are given', async () => {
    // Arrange
    const retryableStreamText = createRetryableStreamText({
      model: new MockLanguageModel(),
      retries: [],
    });

    // Act
    const result = await retryableStreamText({ prompt: 'hi' });
    const direct = streamText({ model: new MockLanguageModel(), prompt: 'hi' });

    // Assert
    expectTypeOf(result).toEqualTypeOf<typeof direct>();
  });
});
