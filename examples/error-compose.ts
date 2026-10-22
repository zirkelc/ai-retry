/**
 * Four ways to write the same error condition.
 *
 * Each entry matches HTTP 503 or 529, or an error message that says the service
 * is unavailable or overloaded. They go from the low-level `error(predicate)`
 * to the high-level `httpStatus()`. Use the shortest form that reads the field
 * you need.
 *
 * Offline: all models are mocks, no API key needed.
 *
 * Run:
 *   pnpm build && pnpm tsx examples/error-compose.ts
 *
 * Output:
 *   retry with anthropic/claude-opus-4-5
 *   text: Why did the robot retry? It had a fallback.
 */
import { APICallError, generateText } from 'ai';
import {
  createRetryableModel,
  error,
  httpStatus,
  or,
} from 'ai-retry/language-model';
import { MockLanguageModel } from 'ai-test-kit/language';

const baseModel = MockLanguageModel.from(
  new APICallError({
    message: 'Service unavailable',
    statusCode: 503,
    url: 'https://api.openai.com/v1/responses',
    requestBodyValues: {},
    isRetryable: true,
  }),
  { provider: 'openai', modelId: 'gpt-5' },
);

const fallbackModel = MockLanguageModel.from(
  'Why did the robot retry? It had a fallback.',
  {
    provider: 'anthropic',
    modelId: 'claude-opus-4-5',
  },
);

const message = /service (unavailable|overloaded)/i;

const retryableModel = createRetryableModel({
  model: baseModel,
  retries: [
    // 1. Low-level predicate with access to the full error
    error((e) => {
      if (!APICallError.isInstance(e)) return false;
      return (
        e.statusCode === 503 || e.statusCode === 529 || message.test(e.message)
      );
    }).switch({ model: fallbackModel }),

    // 2. One matcher for each field
    error.statusCode(503, 529).switch({ model: fallbackModel }),
    error.message(message).switch({ model: fallbackModel }),

    // 3. The two matchers combined into one condition
    or(error.statusCode(503, 529), error.message(message)).switch({
      model: fallbackModel,
    }),

    // 4. The high-level condition: numbers match the status, a regex matches either
    httpStatus(503, 529, message).switch({ model: fallbackModel }),
  ],
  onRetry: ({ current }) => {
    console.log(
      `retry with ${current.model.provider}/${current.model.modelId}`,
    );
  },
});

const result = await generateText({
  model: retryableModel,
  prompt: 'Tell me a joke!',
  maxRetries: 0,
});

console.log(`text: ${result.text}`);
