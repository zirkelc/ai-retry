import { openai } from '@ai-sdk/openai';
import { generateText } from 'ai';
import { createRetryable, timeout } from 'ai-retry/language-model';

/**
 * This example demonstrates using AI SDK's `timeout` option (introduced in v6.0.14)
 * combined with the `timeout()` condition to automatically fall back to a
 * different model when the primary model times out.
 *
 * The `timeout` option accepts milliseconds directly - no AbortSignal needed.
 */
const retryableModel = createRetryable({
  model: openai('gpt-4o'),
  retries: [
    /** Fallback to GPT-4o-mini if the primary model times out */
    timeout().switch({
      model: openai('gpt-4o-mini'),
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
