/**
 * Use Vercel AI Gateway model IDs as strings.
 *
 * A model can be a string. Each entry point resolves the string through the
 * default `gateway` provider, for its own model family. The base model, the
 * fallbacks and the `.switch()` targets can all be strings.
 *
 * Requires: AI_GATEWAY_API_KEY. This example sends real requests.
 *
 * Run:
 *   pnpm build && pnpm tsx examples/ai-gateway.ts
 */
import { embed, gateway, generateText } from 'ai';
import { createRetryableModel as createRetryableEmbeddingModel } from 'ai-retry/embedding-model';
import { createRetryableModel, httpStatus } from 'ai-retry/language-model';

const retryableModel = createRetryableModel({
  model: 'openai/gpt-4.1',
  retries: [
    // A `.switch()` target as a string
    httpStatus(529).switch({ model: 'anthropic/claude-sonnet-4' }),
    // A fallback as a string, and as a gateway model instance
    'google/gemini-2.5-flash',
    gateway('openai/gpt-4.1-mini'),
  ],
});

const retryableEmbeddingModel = createRetryableEmbeddingModel({
  model: 'openai/text-embedding-3-large',
  retries: ['openai/text-embedding-3-small'],
});

const result = await generateText({
  model: retryableModel,
  prompt: 'What is the meaning of life?',
  maxRetries: 0,
});

console.log(`text: ${result.text}`);

const embedding = await embed({
  model: retryableEmbeddingModel,
  value: 'Hello world!',
  maxRetries: 0,
});

console.log(`embedding: ${embedding.embedding.length} dimensions`);
