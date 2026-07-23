/**
 * Prototype: recover a canned refusal at the MODEL layer, with plain
 * `streamText` (no `createRetryableStream`).
 *
 * `createRetryable({ ..., experimental_transform: refusalTransform([...]) })`
 * pipes each attempt's provider stream through a transform that buffers the
 * leading `text-delta`s and, on matching a refusal phrase, emits an `error`
 * part *before any content is forwarded*. That error hits the retryable model's
 * normal pre-content retry path, so an error-based condition fails over to
 * another model — exactly like a provider error. The refusal never reaches the
 * consumer.
 *
 * The transform runs inside `doStream`, so it recovers at the model layer via
 * the existing error conditions — no `createRetryableStream` wrapper needed.
 *
 * Run:
 *   pnpm build && pnpm tsx examples/retryable-model-refusal-transform.ts
 */
import { streamText } from 'ai';
import { createRetryable } from 'ai-retry';
import { error } from 'ai-retry/language-model/conditions';
import {
  RefusalError,
  refusalTransform,
} from 'ai-retry/experimental/transform';
import { Language, MockLanguageModel } from 'ai-test-kit/language';

/** Known false-positive refusals worth failing over from. */
const REFUSALS = [
  "I'm sorry, but I cannot assist",
  'I cannot help with that request',
];

/** A model whose stream is the given text, split into deltas, finishing `stop`. */
const modelStreaming = (deltas: Array<string>) =>
  MockLanguageModel.from({
    doStream: [
      Language.streamStart(),
      ...Language.streamText(deltas, { id: '1' }),
      Language.streamFinish(),
    ],
  });

const scenario = async (label: string, primaryDeltas: Array<string>) => {
  const primary = modelStreaming(primaryDeltas);
  const fallback = modelStreaming([
    'Chlorine tablets ',
    'sanitize pool water.',
  ]);

  /** A bare retryable model with the refusal transform + an error condition. */
  const model = createRetryable({
    model: primary,
    retries: [
      error((e) => e instanceof RefusalError).switch({ model: fallback }),
    ],
    experimental_transform: refusalTransform(REFUSALS),
  });

  /** Plain streamText — no call-layer wrapper. */
  const result = streamText({ model, prompt: 'What sanitizes pool water?' });

  let text = '';
  for await (const delta of result.textStream) text += delta;

  console.log(`\n=== ${label} ===`);
  console.log(`  primary.doStream:  ${primary.doStream.mock.calls.length}`);
  console.log(`  fallback.doStream: ${fallback.doStream.mock.calls.length}`);
  console.log(`  text: ${JSON.stringify(text)}`);
};

/** A canned refusal split across deltas — recovered by failing over. */
await scenario('canned refusal (fail over at the model layer)', [
  "I'm sorry, ",
  'but I cannot assist',
  ' with that request.',
]);

/** A real answer that shares a leading fragment — committed, no false retry. */
await scenario('real answer sharing a fragment (no retry)', [
  "I'm sorry ",
  'to hear that! Chlorine tablets work well.',
]);

console.log(
  '\nTakeaway: the transform converts a soft refusal into a pre-content error,\nso the existing model-layer error conditions recover it under plain streamText.',
);
process.exit(0);
