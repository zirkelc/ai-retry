import { type OpenAIProviderOptions, openai } from '@ai-sdk/openai';
import { generateText } from 'ai';
import { createRetryable } from 'ai-retry';

const retryableModel = createRetryable({
  model: openai('gpt-5'),
  retries: [
    // Use different provider options for the retry
    () => ({
      model: openai('gpt-4o-2024-08-06'),
      providerOptions: {
        openai: {
          user: 'fallback-user',
          textVerbosity: 'high',
        } satisfies OpenAIProviderOptions,
      },
    }),
  ],
});

// Original provider options are used for the first attempt
const result = await generateText({
  model: retryableModel,
  prompt: 'How many "r"s are in the word "strawberry"?',
  providerOptions: {
    openai: {
      user: 'primary-user',
      reasoningEffort: 'high',
    } satisfies OpenAIProviderOptions,
  },
});
