import { azure } from '@ai-sdk/azure';
import { openai } from '@ai-sdk/openai';
import type { LanguageModelV2 } from '@ai-sdk/provider';
import { generateText } from 'ai';
import { createRetryable } from './create-retryable-model.js';
import { contentFilterTriggered } from './retryables/content-filter-triggered.js';
import { requestNotRetryable } from './retryables/request-not-retryable.js';
import { requestTimeout } from './retryables/request-timeout.js';
import { responseSchemaMismatch } from './retryables/response-schema-mismatch.js';

// Create retryable model that will switch models based on error conditions
const retryableModel = createRetryable({
  // Base model to use
  model: azure('gpt-4.1-mini'),
  retries: [
    contentFilterTriggered(openai('gpt-4.1-mini')),
    responseSchemaMismatch(azure('gpt-4.1')),
    requestTimeout(azure('gpt-4.1-mini')),
    requestNotRetryable(azure('gpt-4.1-mini')),
  ],
});

const result = await generateText({
  model: retryableModel,
  prompt: 'Hello world!',
});
