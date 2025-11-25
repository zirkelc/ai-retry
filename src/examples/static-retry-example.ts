import { openai } from '@ai-sdk/openai';
import { generateText } from 'ai';
import { createRetryable } from '../create-retryable-model.js';

// Example using static Retry objects with options
const retryableModel = createRetryable({
  model: openai('gpt-4'),
  retries: [
    // Static Retry object with all options
    {
      model: openai('gpt-4o-mini'),
      maxAttempts: 2,
      delay: 1000,
      providerOptions: {
        openai: {
          user: 'fallback-user',
        },
      },
    },
    // Mix with plain models
    openai('gpt-3.5-turbo'),
    // Mix with functions
    () => ({ model: openai('gpt-4o') }),
  ],
});

// Use the retryable model
async function main() {
  const { text } = await generateText({
    model: retryableModel,
    prompt: 'What is 2+2?',
  });

  console.log(text);
}
