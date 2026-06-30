import { openai } from '@ai-sdk/openai';
import { embed, gateway, generateText } from 'ai';
import { createRetryableModel as createRetryableEmbedding } from '../src/embedding-model/index.js';
import { createRetryableModel } from '../src/language-model/index.js';

const retryableLanguageModel = createRetryableModel({
  model: 'openai/gpt-4.1',
  retries: [
    'anthropic/claude-sonnet-4',
    { model: 'xai/grok-4.3' },
    gateway('openai/gpt-4.1'),
  ],
});

const retryableEmbeddingModel = createRetryableEmbedding({
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
