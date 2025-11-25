import { type GroqProviderOptions, groq } from '@ai-sdk/groq';
import { APICallError, generateText } from 'ai';
import {
  createRetryable,
  isErrorAttempt,
  type LanguageModelV2,
  type Retryable,
} from 'ai-retry';

/**
 * Custom: Flex Tier Capacity Exceeded
 * @see https://console.groq.com/docs/errors
 */
const ERROR_FLEX_TIER_CAPACITY_EXCEEDED = 498;

/**
 * Retryable for Groq's custom 498 error
 */
const groqFlexTierCapacityExceeded: Retryable<LanguageModelV2> = (context) => {
  const { current } = context;

  if (isErrorAttempt(current)) {
    const { error, model } = current;

    if (APICallError.isInstance(error)) {
      const { statusCode } = error;

      if (statusCode === ERROR_FLEX_TIER_CAPACITY_EXCEEDED) {
        // Retry after a delay with exponential backoff
        return { model, maxAttempts: 3, delay: 2_000, backoffFactor: 2 };
      }
    }
  }

  return undefined;
};

const retryableModel = createRetryable({
  model: groq('qwen/qwen3-32b'),
  retries: [groqFlexTierCapacityExceeded],
});

const { text } = await generateText({
  model: retryableModel,
  prompt: 'How many "r"s are in the word "strawberry"?',
});

const growFlexTierCapacityExceeded2: Retryable<LanguageModelV2> = (context) => {
  const { current } = context;

  if (isErrorAttempt(current)) {
    const { error, model } = current;

    if (APICallError.isInstance(error)) {
      const { statusCode } = error;

      if (statusCode === ERROR_FLEX_TIER_CAPACITY_EXCEEDED) {
        // Switch to on-demand tier and lower reasoning effort
        return {
          model: model,
          providerOptions: {
            groq: {
              reasoningEffort: 'low',
              serviceTier: 'on_demand',
            },
          },
        };
      }
    }
  }

  return undefined;
};

const retryableModel2 = createRetryable({
  model: groq('gpt-oss20b/gpt-oss120b'),
  retries: [growFlexTierCapacityExceeded2],
});

const result = await generateText({
  model: retryableModel2,
  providerOptions: {
    groq: {
      // Reasoning effort: low/medium/high
      reasoningEffort: 'high',
      // Service tier: flex/on_demand
      serviceTier: 'abc',
    } satisfies GroqProviderOptions,
  },
  prompt: 'How many "r"s are in the word "strawberry"?',
});
