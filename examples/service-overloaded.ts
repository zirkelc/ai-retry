/**
 * Recover an overloaded provider in a `streamText` call, on the model layer.
 *
 * The retryable model retries a stream before its first content chunk. The
 * base model fails with HTTP 529 before it sends content, so the retryable
 * model retries it, then switches to a different provider.
 *
 * Offline: all models are mocks, no API key needed.
 *
 * Run:
 *   pnpm build && pnpm tsx examples/service-overloaded.ts
 *
 * Output:
 *   retry with anthropic/claude-sonnet-4-5
 *   retry with anthropic/claude-sonnet-4-5
 *   retry with openai/gpt-4o
 *   text: Once upon a time, a robot learned to wait.
 */
import { APICallError, streamText } from 'ai';
import { createRetryableModel, httpStatus } from 'ai-retry/language-model';
import { MockLanguageModel } from 'ai-test-kit/language';

/** Fails with HTTP 529 on every call. */
const baseModel = MockLanguageModel.from(
  new APICallError({
    message: 'Overloaded',
    statusCode: 529,
    url: 'https://api.anthropic.com/v1/messages',
    requestBodyValues: {},
    isRetryable: true,
  }),
  { provider: 'anthropic', modelId: 'claude-sonnet-4-5' },
);

const fallbackModel = MockLanguageModel.from(
  'Once upon a time, a robot learned to wait.',
  {
    provider: 'openai',
    modelId: 'gpt-4o',
  },
);

const retryableModel = createRetryableModel({
  model: baseModel,
  retries: [
    // Retry the same model with exponential backoff
    httpStatus(529, 'overloaded').retry({
      delay: 100,
      backoffFactor: 2,
      maxAttempts: 3,
    }),
    // Then switch to a different provider
    httpStatus(529, 'overloaded').switch({ model: fallbackModel }),
  ],
  onRetry: ({ current }) => {
    console.log(
      `retry with ${current.model.provider}/${current.model.modelId}`,
    );
  },
});

const result = streamText({
  model: retryableModel,
  prompt: 'Write a story about a robot.',
  maxRetries: 0,
});

let text = '';
for await (const chunk of result.textStream) text += chunk;

console.log(`text: ${text}`);
