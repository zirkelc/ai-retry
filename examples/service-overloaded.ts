import { anthropic } from '@ai-sdk/anthropic';
import { openai } from '@ai-sdk/openai';
import { streamText } from 'ai';
import { createRetryable } from 'ai-retry';
import { serviceOverloaded } from 'ai-retry/retryables';

const retryableModel = createRetryable({
  model: anthropic('claude-sonnet-4-0'),
  retries: [
    // Retry with delay and exponential backoff
    serviceOverloaded(anthropic('claude-sonnet-4-0'), {
      delay: 5_000,
      backoffFactor: 2,
      maxAttempts: 5,
    }),
    // Or switch to a different provider
    serviceOverloaded(openai('gpt-4')),
  ],
});

const result = streamText({
  model: retryableModel,
  prompt: 'Write a story about a robot...',
});

for await (const chunk of result.textStream) {
  console.log(chunk);
}
