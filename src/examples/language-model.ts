import { openai } from '@ai-sdk/openai';
import { generateText } from 'ai';
import { createRetryable } from 'ai-retry';
import { retryAfterDelay } from 'ai-retry/retryables';

const retryableModel = createRetryable({
  model: openai('gpt-4'), // Base model
  retries: [
    // Retry base model with fixed 2s delay
    retryAfterDelay({ delay: 2000, maxAttempts: 3 }),

    // Or retry with exponential backoff (2s, 4s, 8s)
    retryAfterDelay({ delay: 2000, maxAttempts: 3, backoffFactor: 2 }),

    // Or switch to a different model after delay
    retryAfterDelay(openai('gpt-4-mini'), { delay: 1000 }),
  ],
});

const { text } = await generateText({
  model: retryableModel,
  prompt: 'What is the meaning of life?',
});

import type { OpenAIResponsesProviderOptions } from '@ai-sdk/openai';

const result = await generateText({
  model: openai('gpt-5'),
  providerOptions: {
    openai: {
      textVerbosity: 'high',
    } satisfies OpenAIResponsesProviderOptions,
  },
  prompt: 'How many "r"s are in the word "strawberry"?',
});
