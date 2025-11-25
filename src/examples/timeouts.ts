import { openai } from '@ai-sdk/openai';
import { generateText } from 'ai';
import { createRetryable } from 'ai-retry';
import { requestTimeout } from 'ai-retry/retryables';

const retryableModel = createRetryable({
  // Base model
  model: openai('gpt-4'),
  retries: [
    // Retry only request timeouts
    requestTimeout(openai('gpt-4-mini'), { timeout: 30_000 }),

    // Or retry any error with a 30 second timeout
    {
      model: openai('gpt-4-mini'),
      timeout: 30_000,
    },
  ],
});

const { text } = await generateText({
  model: retryableModel,
  prompt: 'What is the meaning of life?',
  // Original request timeout
  abortSignal: AbortSignal.timeout(60_000),
});
