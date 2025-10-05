import { anthropic } from '@ai-sdk/anthropic';
import { openai } from '@ai-sdk/openai';
import {
  convertAsyncIterableToArray,
  createTestServer,
} from '@ai-sdk/provider-utils/test';
import { streamText } from 'ai';
import { expect, it } from 'vitest';
import { createRetryable } from './create-retryable-model.js';
import { serviceOverloaded } from './retryables/service-overloaded.js';
import { errorChunk, textChunks } from './test-utils.js';

process.env.ANTHROPIC_API_KEY = 'test-anthropic-key';
process.env.OPENAI_API_KEY = 'test-openai-key';

// Set up test server to mock AI provider responses
const server = createTestServer({
  // Anthropic endpoint
  'https://api.anthropic.com/v1/messages': {
    response: {
      type: 'stream-chunks',
      // Returns overloaded error as first chunk
      chunks: [errorChunk({ type: 'overloaded_error', message: 'Overloaded' })],
    },
  },
  // OpenAI endpoint
  'https://api.openai.com/v1/responses': {
    response: {
      type: 'stream-chunks',
      chunks: [...textChunks.openai(['Hello', ', ', 'World!'])],
    },
  },
});

it('should switch to openai when anthropic is overloaded', async () => {
  const retryableModel = createRetryable({
    // Use Anthropic as base model
    model: anthropic('claude-sonnet-4-20250514'),
    retries: [
      // Switch to OpenAI if Anthropic is overloaded
      serviceOverloaded(openai('gpt-5-2025-08-07')),
    ],
    onError({ current, attempts }) {
      console.error(
        `Error attempt ${attempts.length} from ${current.model.provider}/${current.model.modelId}:`,
        current.error,
      );
    },
    onRetry({ current, attempts }) {
      console.log(
        `Retrying attempt ${attempts.length + 1} with ${current.model.provider}/${current.model.modelId}`,
      );
    },
  });

  const result = streamText({
    // Use retryable model
    model: retryableModel,
    prompt: 'Hello!',
    onChunk({ chunk }) {
      console.log(`Chunk:`, chunk);
    },
    onError(err) {
      // Errors are automatically retried, so this should not be called
      expect.unreachable('Should not log any errors');
    },
  });

  const chunks = await convertAsyncIterableToArray(result.textStream);
  expect(chunks).toEqual(['Hello', ', ', 'World!']);

  const text = await result.text;
  console.log('Final result:', text);
});
