import { anthropic } from '@ai-sdk/anthropic';
import { openai } from '@ai-sdk/openai';
import { APICallError, generateText } from 'ai';
import {
  createRetryable,
  isErrorAttempt,
  type LanguageModelV2,
  type Retryable,
} from 'ai-retry';

const customRetryFunction: Retryable<LanguageModelV2> = (context) => {
  // Context contains the current and all previous attempts
  const { current, attempts } = context;

  if (isErrorAttempt(current)) {
    // Use error and model to decide what to do next
    const { error, model } = current;

    if (APICallError.isInstance(error)) {
      // Implement your custom retry logic here based on the error properties
      const { isRetryable, statusCode, message, data, responseHeaders } = error;

      // Example: Retry with the same model after a delay if the error is retryable
      if (isRetryable) {
        return { model: model, maxAttempts: 3, delay: 1_000 };
      }

      // Example: Switch to a different model for specific status codes or messages
      if (statusCode === 503 || message.includes('service unavailable')) {
        return { model: anthropic('claude-opus-4-0') };
      }
    }
  }

  return undefined;
};

const retryableModel = createRetryable({
  // Base model
  model: openai('gpt-5'),
  retries: [
    // Use your custom retry function
    customRetryFunction,

    // Or use any model as fallback
    anthropic('claude-sonnet-4-0'),
  ],
});

const { text } = await generateText({
  model: retryableModel,
  prompt: 'What is the meaning of life?',
});
