import { openai } from '@ai-sdk/openai';
import { generateObject, generateText } from 'ai';
import { createRetryable, isResultAttempt } from 'ai-retry';
import { requestTimeout, retryAfterDelay } from 'ai-retry/retryables';

const retryableModel = createRetryable({
  // Base model
  model: openai('gpt-4'),
  retries: [
    // sameModel():
    retryAfterDelay({ delay: 30_000, maxAttempts: 3 }), // delay is optional, respects the retry-after header if present
    { model: openai('gpt-4'), maxAttempts: 3 }, // or static fallback model

    // increaseTemperature()
    { model: openai('gpt-4'), temperature: 0.5 }, // possible but NOT implemented yet

    // escalateWithModel()
    { model: openai('gpt-5') }, // static fallback model

    // judgeWithModel()
    // Or retry any error with a 30 second timeout
    async (context) => {
      const { current } = context;

      if (isResultAttempt(current)) {
        const { result } = current;

        if (result.finishReason === 'stop') {
          const content = result.content;

          // Judge with model
          const judging = await generateObject({
            model: openai('gpt-5'),
            prompt: `Judge the following content: ${content}`,
            schema: z.object({
              decision: z.enum(['safe', 'unsafe']),
            }),
          });

          if (judging.object.decision === 'unsafe') {
            return { model: openai('gpt-5'), maxAttempts: 1, ...options };
          }
        }
      }

      return undefined;
    },
  ],
});

const { text } = await generateText({
  model: retryableModel,
  prompt: 'What is the meaning of life?',
  // Original request timeout
  abortSignal: AbortSignal.timeout(60_000),
});
