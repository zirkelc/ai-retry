/**
 * Keep the fallback model sticky for the next requests, with `reset`.
 *
 * By default, each request starts with the base model. With
 * `reset: 'after-2-requests'`, the model of the last successful retry stays for
 * the next 2 requests. Thus the requests do not try the overloaded base model
 * again each time.
 *
 * The other values are `after-request` (the default), `after-N-seconds` and
 * `never`.
 *
 * Offline: all models are mocks, no API key needed.
 *
 * Run:
 *   pnpm build && pnpm tsx examples/reset.ts
 *
 * Output:
 *   request 1: claude-sonnet-4-5, base model calls: 1
 *   request 2: claude-sonnet-4-5, base model calls: 1
 *   request 3: claude-sonnet-4-5, base model calls: 1
 *   request 4: claude-sonnet-4-5, base model calls: 2
 *   request 5: claude-sonnet-4-5, base model calls: 2
 */
import { APICallError, generateText } from 'ai';
import { createRetryableModel, httpStatus } from 'ai-retry/language-model';
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

const fallbackModel = MockLanguageModel.from('Hello!', {
  provider: 'anthropic',
  modelId: 'claude-sonnet-4-5',
});

const retryableModel = createRetryableModel({
  model: baseModel,
  retries: [httpStatus(529).switch({ model: fallbackModel })],
  reset: 'after-2-requests',
});

for (let request = 1; request <= 5; request++) {
  const result = await generateText({
    model: retryableModel,
    prompt: 'Say hello.',
    maxRetries: 0,
  });
  console.log(
    `request ${request}: ${result.response.modelId}, base model calls: ${baseModel.doGenerate.mock.calls.length}`,
  );
}
