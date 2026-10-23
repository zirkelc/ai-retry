/**
 * Retry the same model first, then switch to a different provider.
 *
 * - `error.isRetryable(true).retry()` retries the same model on an error that
 *   the AI SDK marks as retryable, with exponential backoff. A `Retry-After`
 *   response header replaces the delay, up to 60 seconds.
 * - When the retries are used up, `httpStatus(529).switch()` switches to a
 *   different provider.
 *
 * Offline: all models are mocks, no API key needed.
 *
 * Run:
 *   pnpm build && pnpm tsx examples/retry-and-switch.ts
 *
 * Output:
 *   retry with openai/gpt-4o
 *   retry with openai/gpt-4o
 *   retry with anthropic/claude-sonnet-4-5
 *   text: Retries fall like leaves / one model, then another / the answer arrives
 */
import { APICallError, generateText } from 'ai';
import {
  createRetryableModel,
  error,
  httpStatus,
} from 'ai-retry/language-model';
import { MockLanguageModel } from 'ai-test-kit/language';

/** Fails with HTTP 529 on every call. */
const baseModel = MockLanguageModel.from(
  new APICallError({
    message: 'Overloaded',
    statusCode: 529,
    url: 'https://api.openai.com/v1/responses',
    requestBodyValues: {},
    isRetryable: true,
  }),
  { provider: 'openai', modelId: 'gpt-4o' },
);

const fallbackModel = MockLanguageModel.from(
  'Retries fall like leaves / one model, then another / the answer arrives',
  {
    provider: 'anthropic',
    modelId: 'claude-sonnet-4-5',
  },
);

const retryableModel = createRetryableModel({
  model: baseModel,
  retries: [
    // Retry the same model up to 3 attempts, after 100 ms and 200 ms
    error
      .isRetryable(true)
      .retry({ delay: 100, backoffFactor: 2, maxAttempts: 3 }),

    // Then switch to a different provider on overload
    httpStatus(529, 'overloaded').switch({ model: fallbackModel }),
  ],
  onRetry: ({ current }) => {
    console.log(
      `retry with ${current.model.provider}/${current.model.modelId}`,
    );
  },
});

const result = await generateText({
  model: retryableModel,
  prompt: 'Write a haiku about retries.',
  maxRetries: 0,
});

console.log(`text: ${result.text}`);
