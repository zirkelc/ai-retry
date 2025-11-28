import { openai } from '@ai-sdk/openai';
import { gateway, generateText } from 'ai';
import { createRetryable, EmbeddingModel } from '../src/index.js';

const retryableLanguageModelDefault = createRetryable({
  model: 'openai/gpt-4.1',
  retries: [
    'anthropic/claude-sonnet-4',
    { model: 'xai/grok-4' },
    gateway('openai/gpt-4.1'),
  ],
});

const retryableLanguageModel = createRetryable({
  model: gateway('openai/gpt-4.1'),
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
