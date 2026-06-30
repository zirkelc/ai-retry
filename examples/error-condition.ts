import { anthropic } from '@ai-sdk/anthropic';
import { openai } from '@ai-sdk/openai';
import { APICallError, generateText } from 'ai';
import {
  createRetryableModel,
  error,
  timeout,
} from '../src/language-model/index.js';

/**
 * The `error(predicate)` condition takes an arbitrary predicate over the
 * failed attempt's error. Reach for it whenever the higher-level helpers
 * (`httpStatus`, `error.statusCode`, `error.message`, `error.isRetryable`)
 * do not cover the field you need.
 */
const retryableModel = createRetryableModel({
  model: openai('gpt-4o'),
  retries: [
    timeout().switch({
      model: openai('gpt-4o-mini'),
      timeout: 30_000, // 30 second timeout for the fallback
    }),

    // 1. OpenAI-style error code, nested at `data.error.code`
    error((e, ctx) => {
      if (!APICallError.isInstance(e)) return false;
      const data = e.data as { error?: { code?: string } } | undefined;
      return data?.error?.code === 'content_filter';
    }).switch({ model: anthropic('claude-3-haiku-20240307') }),

    // // 2. Anthropic-style error envelope, top-level `data.type`
    // error((e) => {
    //   if (!APICallError.isInstance(e)) return false;
    //   const data = e.data as { type?: string } | undefined;
    //   return data?.type === 'overloaded_error';
    // }).switch({ model: openai('gpt-4o-mini') }),

    // // 3. Regex match against the error message
    // error(
    //   (e) => e instanceof Error && /quota.*exceeded/i.test(e.message),
    // ).switch({ model: anthropic('claude-3-haiku-20240307') }),

    // // 4. Inspect a custom response header
    // error((e) => {
    //   if (!APICallError.isInstance(e)) return false;
    //   return e.responseHeaders?.['x-fallback'] === 'true';
    // }).switch({ model: openai('gpt-4o-mini') }),
  ],
});

const result = await generateText({
  model: retryableModel,
  prompt: 'Hello, world!',
});

console.log(result.text);
