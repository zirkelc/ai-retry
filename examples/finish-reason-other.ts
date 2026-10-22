/**
 * Retry the same model when it finishes with the reason `other`.
 *
 * `finishReason()` is a result condition. It examines a successful response, so
 * it can retry a response that did not fail but is not usable.
 *
 * Offline: all models are mocks, no API key needed.
 *
 * Run:
 *   pnpm build && pnpm tsx examples/finish-reason-other.ts
 *
 * Output:
 *   retry after finish reason other
 *   text: The meaning of life is 42.
 *   finish reason: stop
 */
import { generateText } from 'ai';
import { isResultAttempt } from 'ai-retry';
import { createRetryableModel, finishReason } from 'ai-retry/language-model';
import { Language, MockLanguageModel } from 'ai-test-kit/language';

/** The first response finishes with `other`, the second with `stop`. */
const baseModel = MockLanguageModel.from(
  [
    { content: [Language.text('The meaning of')], finishReason: 'other' },
    'The meaning of life is 42.',
  ],
  { provider: 'openai', modelId: 'gpt-4o' },
);

const retryableModel = createRetryableModel({
  model: baseModel,
  retries: [finishReason('other').retry({ maxAttempts: 3 })],
  onRetry: ({ current }) => {
    if (isResultAttempt(current)) {
      console.log(`retry after finish reason ${current.finishReason}`);
    }
  },
});

const result = await generateText({
  model: retryableModel,
  prompt: 'What is the meaning of life?',
  maxRetries: 0,
});

console.log(`text: ${result.text}`);
console.log(`finish reason: ${result.finishReason}`);
