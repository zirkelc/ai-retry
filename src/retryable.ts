import { azure } from '@ai-sdk/azure';
import { openai } from '@ai-sdk/openai';
import type { LanguageModelV2 } from '@ai-sdk/provider';
import { generateText } from 'ai';
import { createRetryable } from './create-retryable-model.js';

// Create retryable model that will switch models based on error conditions
const retryableModel = createRetryable({
  // Base model to use
  model: azure('gpt-4.1-mini'),
  retries: [
    // If content filter was triggered, switch provider Azure -> OpenAI
    finishReasonContentFilter(openai('gpt-4.1-mini')),
    // If structured output validation failed, switch to more powerful model gpt-4.1-mini -> gpt-4.1
    typeValidationError(azure('gpt-4.1')),
    // In all other cases, retry the base model once again
    retryOnce(),
  ],
});

const result = await generateText({
  model: retryableModel,
  prompt: 'Hello world!',
});

declare function finishReasonContentFilter(
  model: LanguageModelV2,
): LanguageModelV2;

declare function typeValidationError(model: LanguageModelV2): LanguageModelV2;

declare function retry(
  model: LanguageModelV2,
  { retries }: { retries: number },
): LanguageModelV2;

declare function retryOnce(): LanguageModelV2;
