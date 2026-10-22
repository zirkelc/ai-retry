/**
 * Retry with a higher temperature until a judge model rates the text as creative.
 *
 * A retryable is a function that gets the retry context and returns a `Retry`
 * or `undefined`. A hand-written retryable can compute the next attempt from
 * the current attempt. Here, each retry increases the temperature by 0.2.
 *
 * Offline: all models are mocks, no API key needed.
 *
 * Run:
 *   pnpm build && pnpm tsx examples/creativity-checker.ts
 *
 * Output:
 *   score 3 at temperature 0.3
 *   score 8 at temperature 0.5
 *   text: The lighthouse robot collected lost echoes and played them back to ships.
 */
import { generateText, Output } from 'ai';
import {
  isResultAttempt,
  type LanguageModel,
  type ModelRetryable,
} from 'ai-retry';
import { createRetryableModel } from 'ai-retry/language-model';
import { MockLanguageModel } from 'ai-test-kit/language';
import { z } from 'zod';

/** The first story is generic, the second is creative. */
const storyModel = MockLanguageModel.from(
  [
    'A robot woke up and did its job.',
    'The lighthouse robot collected lost echoes and played them back to ships.',
  ],
  { provider: 'openai', modelId: 'gpt-4.1' },
);

/** Rates the first story 3, and the second story 8. */
const judgeModel = MockLanguageModel.from(
  [JSON.stringify({ score: 3 }), JSON.stringify({ score: 8 })],
  { provider: 'openai', modelId: 'gpt-4.1-mini' },
);

const creativityChecker: ModelRetryable<LanguageModel> = async (context) => {
  const { current } = context;

  // Retryables run for errors and for results. This one checks results only.
  if (!isResultAttempt(current)) return undefined;
  if (current.finishReason !== 'stop') return undefined;

  const text = current.result.content
    .filter((part) => part.type === 'text')
    .map((part) => part.text)
    .join('\n');

  const { output } = await generateText({
    model: judgeModel,
    output: Output.object({
      schema: z.object({ score: z.number().min(1).max(10) }),
    }),
    prompt: `Rate this text for creativity from 1 to 10:\n\n${text}`,
  });

  console.log(
    `score ${output.score} at temperature ${current.options.temperature}`,
  );
  if (output.score >= 5) return undefined;

  const temperature = current.options.temperature ?? 0.5;
  if (temperature >= 1) return undefined;

  // Retry the same model with a higher temperature
  return {
    model: current.model,
    options: { temperature: Math.min(temperature + 0.2, 1) },
    maxAttempts: 5,
  };
};

const retryableModel = createRetryableModel({
  model: storyModel,
  retries: [creativityChecker],
});

const result = await generateText({
  model: retryableModel,
  prompt: 'Write a short story about a robot.',
  temperature: 0.3,
  maxRetries: 0,
});

console.log(`text: ${result.text}`);
