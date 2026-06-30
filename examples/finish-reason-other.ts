import { openai } from '@ai-sdk/openai';
import { generateText } from 'ai';
import { isResultAttempt, type LanguageModel, type Retryable } from 'ai-retry';
import { createRetryableModel } from 'ai-retry/language-model';

const finishReasonOther: Retryable<LanguageModel> = async (context) => {
  // Current attempt: error or result
  const { current } = context;

  // Retryables are called for errors and results
  // We can check the current attempt to see if it's an error or a result
  if (isResultAttempt(current)) {
    // Result comes from generateText or generateObject
    const { result, model } = current;

    // The model finished with an other reason
    if (result.finishReason.unified === 'other') {
      // Retry with the base model 3 times
      return { model, maxAttempts: 3 };
    }
  }

  // Skip to next retryable or finish
  return undefined;
};

// Create a retryable model with the guardrail
const retryableModel = createRetryableModel({
  model: openai('gpt-4'), // Base model
  retries: [finishReasonOther],
});

// Use like any other AI SDK model
const result = await generateText({
  model: retryableModel,
  prompt: 'What is the meaning of life?',
});
