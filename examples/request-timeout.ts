import { openai } from '@ai-sdk/openai';
import { generateText } from 'ai';
import { createRetryable } from 'ai-retry';
import { requestTimeout } from '../src/retryables/request-timeout.js';

/**
 * This example demonstrates using AI SDK's `timeout` option (introduced in v6.0.14)
 * combined with the `requestTimeout` retryable to automatically fallback to a
 * different model when the primary model times out.
 *
 * The `timeout` option accepts milliseconds directly - no AbortSignal needed.
 */
const retryableModel = createRetryable({
  model: openai('gpt-4o'),
  retries: [
    /** Fallback to GPT-4o if the primary model times out */
    requestTimeout(openai('gpt-4o-mini'), {
      timeout: 30_000, // 30 second timeout for the fallback
    }),
  ],
});

const result = await generateText({
  model: retryableModel,
  prompt: `Invent a new holiday and describe its traditions.`,
  timeout: 5_000, // 5 second timeout (AI SDK 6.0.14+)
});

console.log(result.text);
