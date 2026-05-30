import { anthropic } from '@ai-sdk/anthropic';
import { openai } from '@ai-sdk/openai';
import { generateText } from 'ai';
import { createRetryable, error, httpStatus } from 'ai-retry/language-model';

/**
 * Combine `.retry()` and `.switch()` on the same retryable model.
 *
 * - Transient errors (anything the AI SDK marks retryable, including 5xx and
 *   network errors) get one extra attempt on the same model with exponential
 *   backoff. If the response carries a `Retry-After` header, it overrides
 *   the configured delay (capped at 60 seconds).
 * - HTTP 529 (overloaded) keeps the same retry on the primary, but then
 *   falls back to a different provider if the primary still fails.
 */
const retryableModel = createRetryable({
  model: openai('gpt-4o'),
  retries: [
    // Retry the same model on transient errors
    error
      .isRetryable(true)
      .retry({ delay: 1_000, backoffFactor: 2, maxAttempts: 3 }),

    // After that, fall back to a different provider on overload
    httpStatus(529, 'overloaded').switch({
      model: anthropic('claude-sonnet-4-0'),
    }),
  ],
});

const result = await generateText({
  model: retryableModel,
  prompt: 'Write a haiku about retries.',
});

console.log(result.text);
