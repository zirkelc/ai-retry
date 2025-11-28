import { anthropic } from '@ai-sdk/anthropic';
import { openai } from '@ai-sdk/openai';
import { generateObject, generateText } from 'ai';
import type { LanguageModel, Retryable } from 'ai-retry';
import { createRetryable, isResultAttempt } from 'ai-retry';
import { z } from 'zod';

const guardrail: Retryable<LanguageModel> = async (context) => {
  // Current attempt: error or result
  const { current } = context;

  // Retryables are called for errors and results
  // We can check the current attempt to see if it's an error or a result
  if (isResultAttempt(current)) {
    // Result comes from generateText or generateObject
    const { result } = current;

    // If the model finished the generation, we can check the content
    if (result.finishReason === 'stop') {
      const content = result.content
        .filter((c) => c.type === 'text')
        .map((c) => c.text)
        .join('\n');

      // Use a small model to judge the content for safety
      const { object } = await generateObject({
        model: openai('gpt-4.1-mini'),
        prompt: `
          You are a helpful assistant that determines if the generated content is safe for work.
          Content: ${content}
        `,
        schema: z.object({
          safe: z.boolean(),
        }),
      });

      // If the content is not safe, retry with a different model
      if (object.safe) {
        return { model: anthropic('claude-sonnet-4-0') };
      }
    }
  }

  // Skip to next retryable or finish
  return undefined;
};

// Create a retryable model with the guardrail
const retryableModel = createRetryable({
  model: openai('gpt-4'), // Base model
  retries: [guardrail],
});

// Use like any other AI SDK model
const result = await generateText({
  model: retryableModel,
  prompt: 'Tell me a dirty joke',
});
