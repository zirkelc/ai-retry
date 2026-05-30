import { anthropic } from '@ai-sdk/anthropic';
import { openai } from '@ai-sdk/openai';
import { APICallError, generateText } from 'ai';
import {
  createRetryable,
  error,
  httpStatus,
  or,
} from '../src/language-model/index.js';

const baseModel = openai('gpt-5');
const fallbackModel = anthropic('claude-opus-4-5');

const retryableModel = createRetryable({
  model: baseModel,
  retries: [
    // Low-level error condition with access to full error and retry context
    error((e, ctx) => {
      if (APICallError.isInstance(e)) {
        if (e.statusCode === 503 || e.statusCode === 529) return true;
        if (
          e.message.includes('service unavilable') ||
          e.message.includes('service overloaded')
        )
          return true;
      }
      return false;
    }).switch({ model: fallbackModel }),

    // Same as single conditions:
    error.statusCode(503, 529).switch({ model: fallbackModel }), // number or regex
    error
      .message(/service (unavilable|overloaded)/i)
      .switch({ model: fallbackModel }), // string or regex

    // Same as one combined condition:
    or(
      error.statusCode(503, 529),
      error.message(/service (unavilable|overloaded)/i),
    ).switch({ model: fallbackModel }),

    // Or composed into a high-level condition:
    httpStatus(503, 529, /service (unavilable|overloaded)/i).switch({
      model: fallbackModel,
    }), // number, string or regex
  ],
});

const result = await generateText({
  model: retryableModel,
  prompt: 'Tell me a joke!',
});

console.log(result.text);
