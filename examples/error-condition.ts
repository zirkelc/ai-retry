/**
 * Match a field that no helper reads, with `error(predicate)`.
 *
 * `error(predicate)` takes any predicate over the error of the failed attempt.
 * Use it when `httpStatus()`, `error.statusCode()`, `error.message()` and
 * `error.isRetryable()` do not read the field that you need. Here, the field is
 * an OpenAI-style error code at `data.error.code`.
 *
 * Offline: all models are mocks, no API key needed.
 *
 * Run:
 *   pnpm build && pnpm tsx examples/error-condition.ts
 *
 * Output:
 *   retry with anthropic/claude-haiku-4-5
 *   text: Hello! How can I help?
 */
import { APICallError, generateText } from 'ai';
import { createRetryableModel, error } from 'ai-retry/language-model';
import { MockLanguageModel } from 'ai-test-kit/language';

const baseModel = MockLanguageModel.from(
  new APICallError({
    message: 'The response was filtered',
    statusCode: 400,
    url: 'https://api.openai.com/v1/responses',
    requestBodyValues: {},
    data: { error: { code: 'content_filter' } },
    isRetryable: false,
  }),
  { provider: 'openai', modelId: 'gpt-4o' },
);

const fallbackModel = MockLanguageModel.from('Hello! How can I help?', {
  provider: 'anthropic',
  modelId: 'claude-haiku-4-5',
});

/** `e` is `unknown`, so check the error class first. */
const isContentFilter = (e: unknown) => {
  if (!APICallError.isInstance(e)) return false;
  const data = e.data as { error?: { code?: string } } | undefined;
  return data?.error?.code === 'content_filter';
};

const retryableModel = createRetryableModel({
  model: baseModel,
  retries: [error(isContentFilter).switch({ model: fallbackModel })],
  onRetry: ({ current }) => {
    console.log(
      `retry with ${current.model.provider}/${current.model.modelId}`,
    );
  },
});

const result = await generateText({
  model: retryableModel,
  prompt: 'Hello, world!',
  maxRetries: 0,
});

console.log(`text: ${result.text}`);
