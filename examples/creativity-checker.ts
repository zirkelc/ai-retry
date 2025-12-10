import { anthropic } from '@ai-sdk/anthropic';
import { openai } from '@ai-sdk/openai';
import { generateObject, generateText } from 'ai';
import type { LanguageModel, Retryable } from 'ai-retry';
import { createRetryable, isResultAttempt } from 'ai-retry';
import { z } from 'zod';

const creativityChecker: Retryable<LanguageModel> = async (context) => {
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

      // Use a small model to judge creativity
      const { object } = await generateObject({
        model: openai('gpt-4.1-mini'),
        prompt: `
          You are a creativity judge.
          Rate the following text for creativity on a scale of 1-10.
          A score below 5 means the text is generic, predictable, or boring.
          A score of 5 or above means the text is creative, unique, or interesting.

          Text to evaluate:
          ${content}
        `,
        schema: z.object({
          score: z.number().min(1).max(10),
        }),
      });

      // If the content is not creative enough, retry with higher temperature
      if (object.score < 5) {
        // Get the temperature from the current attempt, default to 0.5
        const currentTemperature = current.options.temperature ?? 0.5;

        // Increase temperature by 0.1
        const newTemperature = currentTemperature + 0.1;

        // If we've maxed out temperature, give up
        if (currentTemperature >= 1.0) {
          return undefined;
        }

        // Retry with the same model but higher temperature
        return {
          model: current.model,
          options: {
            temperature: newTemperature,
          },
          maxAttempts: 5, // Stop at 5 attempts
        };
      }
    }
  }

  // Skip to next retryable or finish
  return undefined;
};

// Create a retryable model with the creativity checker
const retryableModel = createRetryable({
  model: anthropic('claude-sonnet-4-20250514'), // Base model
  retries: [creativityChecker],
});

// Use like any other AI SDK model
const result = await generateText({
  model: retryableModel,
  prompt: 'Write a short story about a robot.',
  temperature: 0.3, // Start with low temperature
});
