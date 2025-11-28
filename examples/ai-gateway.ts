import { openai } from '@ai-sdk/openai';
import { embed, gateway, generateText } from 'ai';
import { createRetryable, EmbeddingModel } from '../src/index.js';

const retryableLanguageModel = createRetryable({
  model: 'openai/gpt-4.1',
  retries: [
    'anthropic/claude-sonnet-4',
    { model: 'xai/grok-4' },
    gateway('openai/gpt-4.1'),
  ],
});

const retryableEmbeddingModel = createRetryable({
  model: gateway.textEmbeddingModel('openai/text-embedding-3-large'),
  retries: [],
});

const result = await generateText({
  model: retryableLanguageModel,
  prompt: 'What is the meaning of life?',
});

console.log(result.text);

const embeddingResult = await embed({
  model: retryableEmbeddingModel,
  value: 'Hello world!',
});

console.log(embeddingResult.embedding);
