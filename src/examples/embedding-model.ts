import { azure } from '@ai-sdk/azure';
import { openai } from '@ai-sdk/openai';
import { embed } from 'ai';
import { createRetryable } from 'ai-retry';

// Create a retryable model
const retryableModel = createRetryable({
  // Use OpenAI as base model
  model: openai.textEmbedding('text-embedding-3-large'),
  retries: [
    // Fallback to Azure on failure
    azure.textEmbedding('text-embedding-3-large'),
  ],
});

const { embedding } = await embed({
  model: retryableModel,
  value: 'Hello world!',
  // Timeout after 60 seconds
  abortSignal: AbortSignal.timeout(60_000),
});

console.log(embedding);
