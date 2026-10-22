/**
 * Switch to a different model when a judge model rates the result as unsafe.
 *
 * `result(predicate)` is a result condition. Its predicate can be `async`, so it
 * can call a second model to judge the result. When the predicate returns
 * `true`, the condition switches to the fallback model.
 *
 * Offline: all models are mocks, no API key needed.
 *
 * Run:
 *   pnpm build && pnpm tsx examples/guardrail.ts
 *
 * Output:
 *   judge: unsafe
 *   judge: safe
 *   text: Why do programmers prefer dark mode? Because light attracts bugs.
 */
import { generateText, Output } from 'ai';
import { createRetryableModel, result } from 'ai-retry/language-model';
import { MockLanguageModel } from 'ai-test-kit/language';
import { z } from 'zod';

const baseModel = MockLanguageModel.from('A joke that is not safe for work.', {
  provider: 'openai',
  modelId: 'gpt-4o',
});

const fallbackModel = MockLanguageModel.from(
  'Why do programmers prefer dark mode? Because light attracts bugs.',
  { provider: 'anthropic', modelId: 'claude-sonnet-4-5' },
);

/** Rates the first text as unsafe, and the second as safe. */
const judgeModel = MockLanguageModel.from(
  [JSON.stringify({ safe: false }), JSON.stringify({ safe: true })],
  { provider: 'openai', modelId: 'gpt-4.1-mini' },
);

/** Asks the judge model if the text is safe for work. */
const isUnsafe = async (text: string) => {
  const { output } = await generateText({
    model: judgeModel,
    output: Output.object({ schema: z.object({ safe: z.boolean() }) }),
    prompt: `Is this text safe for work?\n\n${text}`,
  });
  console.log(`judge: ${output.safe ? 'safe' : 'unsafe'}`);
  return !output.safe;
};

const retryableModel = createRetryableModel({
  model: baseModel,
  retries: [
    result((res) => {
      const text = res.content
        .filter((part) => part.type === 'text')
        .map((part) => part.text)
        .join('\n');
      return isUnsafe(text);
    }).switch({ model: fallbackModel }),
  ],
});

const response = await generateText({
  model: retryableModel,
  prompt: 'Tell me a joke.',
  maxRetries: 0,
});

console.log(`text: ${response.text}`);
