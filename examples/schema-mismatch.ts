import { openai } from '@ai-sdk/openai';
import type { LanguageModelV3 } from '@ai-sdk/provider';
import { generateText, Output } from 'ai';
import { createRetryable } from 'ai-retry';
import { schemaMismatch } from 'ai-retry/retryables';
import { z } from 'zod';

/**
 * Creates a mock language model that returns a fixed JSON string.
 */
function createMockModel(modelId: string, response: string): LanguageModelV3 {
  return {
    specificationVersion: `v3`,
    provider: `mock`,
    modelId,
    supportedUrls: {},
    doGenerate: async () => ({
      finishReason: { unified: `stop`, raw: undefined },
      usage: {
        inputTokens: { total: 10, noCache: 0, cacheRead: 0, cacheWrite: 0 },
        outputTokens: { total: 20, text: 0, reasoning: 0 },
      },
      content: [{ type: `text`, text: response }],
      warnings: [],
    }),
    doStream: async () => {
      throw new Error(`Not implemented`);
    },
  };
}

/**
 * Primary model returns an enum value outside the allowed set.
 * Weaker models often paraphrase instead of picking an exact enum value.
 */
const primaryModel = createMockModel(
  `weak-model`,
  JSON.stringify({ sentiment: `kind of positive` }),
);

/**
 * Fallback model returns a valid enum value.
 */
const fallbackModel = createMockModel(
  `strong-model`,
  JSON.stringify({ sentiment: `positive` }),
);

const retryableModel = createRetryable({
  // Weaker base model
  model: primaryModel,
  // model: openai('gpt-4.1-nano'),
  retries: [
    // Retry with a stronger model
    schemaMismatch(fallbackModel),
    // schemaMismatch(openai('gpt-5-pro')),
  ],
  onRetry: ({ current }) => {
    console.log(`Retrying with model: ${current.model.modelId}`);
  },
});

const result = await generateText({
  model: retryableModel,
  maxRetries: 0,
  output: Output.object({
    schema: z.object({
      sentiment: z.enum([`positive`, `negative`, `neutral`]),
    }),
  }),
  prompt: `Classify the sentiment of: "I really enjoyed this movie!"`,
});

console.log(`Result:`, result.output);
// Output:
// Model response did not match schema: {"sentiment":"kind of positive"}
// Retrying with model: strong-model
// Result: { sentiment: 'positive' }
