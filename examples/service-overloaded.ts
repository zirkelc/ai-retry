import { anthropic } from '@ai-sdk/anthropic';
import { openai } from '@ai-sdk/openai';
import { streamText } from 'ai';
import { createRetryableModel, httpStatus } from 'ai-retry/language-model';

const retryableModel = createRetryableModel({
  model: anthropic('claude-sonnet-4-0'),
  retries: [
    // Retry the same model with delay and exponential backoff
    httpStatus(529, 'overloaded').retry({
      delay: 5_000,
      backoffFactor: 2,
      maxAttempts: 5,
    }),
    // Or switch to a different provider
    httpStatus(529, 'overloaded').switch({ model: openai('gpt-4') }),
  ],
});

const result = streamText({
  model: retryableModel,
  prompt: 'Write a story about a robot...',
});

for await (const chunk of result.textStream) {
  console.log(chunk);
}
