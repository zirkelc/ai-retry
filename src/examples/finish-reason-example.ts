import { openai } from '@ai-sdk/openai';
import { generateText, streamText } from 'ai';
import { createRetryable, finishReason } from 'ai-retry';

// Example 1: Retry on unknown finish reasons with generateText
const model1 = createRetryable({
  model: openai('gpt-4'),
  retries: [
    // If the model finishes with 'unknown' reason, retry with fallback
    finishReason(openai('gpt-4-mini'), {
      reasons: 'unknown',
      maxAttempts: 1,
    }),
  ],
});

const result1 = await generateText({
  model: model1,
  prompt: 'What is the meaning of life?',
});

console.log(result1.text);

// Example 2: Retry on multiple finish reasons
const model2 = createRetryable({
  model: openai('gpt-4'),
  retries: [
    // Retry on multiple problematic finish reasons
    finishReason(openai('gpt-4-turbo'), {
      reasons: ['unknown', 'error', 'other'],
      maxAttempts: 2,
      delay: 1000,
    }),
  ],
});

const result2 = await generateText({
  model: model2,
  prompt: 'Explain quantum computing',
});

console.log(result2.text);

// Example 3: Works with streamText when no content has been streamed
const model3 = createRetryable({
  model: openai('gpt-4'),
  retries: [
    finishReason(openai('gpt-4-mini'), {
      reasons: ['unknown', 'error'],
      maxAttempts: 1,
    }),
  ],
});

const stream = streamText({
  model: model3,
  prompt: 'Write a haiku about coding',
});

// If the first model fails with unknown/error finish reason BEFORE streaming any content,
// it will automatically retry with gpt-4-mini
for await (const chunk of stream.textStream) {
  process.stdout.write(chunk);
}
