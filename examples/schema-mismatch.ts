/**
 * Switch to a stronger model when the result does not match the output schema.
 *
 * `schemaInvalid()` matches when the result text fails JSON-schema validation
 * against the `responseFormat` that `Output.object()` sets. Weaker models often
 * write a free value where the schema has an enum.
 *
 * Offline: all models are mocks, no API key needed.
 *
 * Run:
 *   pnpm build && pnpm tsx examples/schema-mismatch.ts
 *
 * Output:
 *   retry with openai/gpt-4.1
 *   output: {"sentiment":"positive"}
 */
import { generateText, Output } from 'ai';
import { createRetryableModel, schemaInvalid } from 'ai-retry/language-model';
import { MockLanguageModel } from 'ai-test-kit/language';
import { z } from 'zod';

/** Returns a value that is not in the enum. */
const baseModel = MockLanguageModel.from(
  JSON.stringify({ sentiment: 'kind of positive' }),
  { provider: 'openai', modelId: 'gpt-4.1-nano' },
);

/** Returns a valid enum value. */
const fallbackModel = MockLanguageModel.from(
  JSON.stringify({ sentiment: 'positive' }),
  { provider: 'openai', modelId: 'gpt-4.1' },
);

const retryableModel = createRetryableModel({
  model: baseModel,
  retries: [schemaInvalid().switch({ model: fallbackModel })],
  onRetry: ({ current }) => {
    console.log(
      `retry with ${current.model.provider}/${current.model.modelId}`,
    );
  },
});

const result = await generateText({
  model: retryableModel,
  output: Output.object({
    schema: z.object({
      sentiment: z.enum(['positive', 'negative', 'neutral']),
    }),
  }),
  prompt: 'Classify the sentiment of: "I really enjoyed this movie!"',
  maxRetries: 0,
});

console.log(`output: ${JSON.stringify(result.output)}`);
