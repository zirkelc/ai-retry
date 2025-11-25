import { anthropic } from '@ai-sdk/anthropic';
import { openai } from '@ai-sdk/openai';
import { generateText } from 'ai';
import { createRetryable } from 'ai-retry';
import { requestNotRetryable } from 'ai-retry/retryables';

const retryableModel = createRetryable({
  model: openai('gpt-4'),
  retries: [
    requestNotRetryable(openai('gpt-4-mini')),

    anthropic('claude-opus-4-20250514'),
  ],
  // Disable retries in test environment
  disabled: process.env.NODE_ENV === 'test',
  // Or check feature flags at runtime
  // disabled: () => !featureFlags.isEnabled('ai-retries'),
});

// Use like any other model
const { text } = await generateText({
  model: retryableModel,
  prompt: 'What is the meaning of life?',
});

console.log(text);
