import { openai } from '@ai-sdk/openai';
import type { LanguageModelV3 } from '@ai-sdk/provider';
import { APICallError, generateText } from 'ai';
import { createRetryable } from 'ai-retry';
import { serviceOverloaded } from 'ai-retry/retryables';

/**
 * Creates a mock language model that always throws a 529 (overloaded) error.
 */
function createOverloadedModel(modelId: string): LanguageModelV3 {
  return {
    specificationVersion: `v3`,
    provider: `mock`,
    modelId,
    supportedUrls: {},
    doGenerate: async () => {
      throw new APICallError({
        message: `Service overloaded`,
        url: `https://api.example.com/v1/chat/completions`,
        requestBodyValues: {},
        statusCode: 529,
        isRetryable: true,
      });
    },
    doStream: async () => {
      throw new Error(`Not implemented`);
    },
  };
}

/**
 * Creates a mock language model that always succeeds.
 */
function createSuccessModel(modelId: string): LanguageModelV3 {
  let callCount = 0;
  return {
    specificationVersion: `v3`,
    provider: `mock`,
    modelId,
    supportedUrls: {},
    doGenerate: async () => {
      callCount++;
      console.log(`  → ${modelId} handled request (call #${callCount})`);
      return {
        finishReason: { unified: `stop`, raw: undefined },
        usage: {
          inputTokens: { total: 10, noCache: 0, cacheRead: 0, cacheWrite: 0 },
          outputTokens: { total: 20, text: 0, reasoning: 0 },
        },
        content: [{ type: `text`, text: `Hello from ${modelId}` }],
        warnings: [],
      };
    },
    doStream: async () => {
      throw new Error(`Not implemented`);
    },
  };
}

/**
 * Primary model is overloaded — every call throws a 529 error.
 * Fallback model always succeeds.
 *
 * With `reset: 'after-2-requests'`, the fallback stays sticky for 2 more
 * requests after a retry, avoiding unnecessary failed calls to the
 * overloaded primary.
 */
const primaryModel = createOverloadedModel(`primary-model`);
const fallbackModel = createSuccessModel(`fallback-model`);

const retryableModel = createRetryable({
  // model: openai('gpt-4o-mini'),
  model: primaryModel,
  retries: [
    // Switch to the fallback model on service overloads (HTTP 529)
    serviceOverloaded(fallbackModel),
    // serviceOverloaded(fallbackModel)
  ],
  // Reset policy:
  // - after-request (default): every new request starts with the base model
  // - after-N-requests: retry model stays sticky for the next N requests
  // - after-N-seconds: retry model stays sticky for N seconds
  reset: `after-2-requests`,
  onRetry: ({ current }) => {
    console.log(`  ⟳ retrying with: ${current.model.modelId}`);
  },
});

for (let i = 1; i <= 5; i++) {
  console.log(`\nRequest ${i}:`);
  const result = await generateText({
    model: retryableModel,
    maxRetries: 0,
    prompt: `Say hello`,
  });
  console.log(`  Result: ${result.text}`);
}

// Output:
//
// Request 1:
//   ⟳ retrying with: fallback-model
//   → fallback-model handled request (call #1)
//   Result: Hello from fallback-model
//
// Request 2:                                    ← sticky (1 of 2)
//   → fallback-model handled request (call #2)
//   Result: Hello from fallback-model
//
// Request 3:                                    ← sticky (2 of 2)
//   → fallback-model handled request (call #3)
//   Result: Hello from fallback-model
//
// Request 4:                                    ← reset, primary fails again
//   ⟳ retrying with: fallback-model
//   → fallback-model handled request (call #4)
//   Result: Hello from fallback-model
//
// Request 5:                                    ← sticky (1 of 2)
//   → fallback-model handled request (call #5)
//   Result: Hello from fallback-model
