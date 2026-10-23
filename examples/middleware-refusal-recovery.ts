/**
 * Switch to a fallback model when the model answers with a refusal, with a middleware.
 *
 * A refusal such as "I'm sorry, but I cannot assist with that request." is
 * normal text that finishes with `stop`. It is not an error, and its finish
 * reason is not `content-filter`. A `wrapLanguageModel` middleware holds back
 * the first stream parts. When the text matches a refusal, it errors the stream
 * with a `RefusalError` before any part reaches the consumer.
 *
 * `controller.error()` has two effects:
 *
 * 1. `pipeThrough` cancels the source stream, so the request to the provider
 *    stops and uses no more tokens.
 * 2. The read of the consumer rejects with the `RefusalError`. The retryable
 *    model gets it before content, so `error.isInstance(RefusalError)` can
 *    switch to the fallback model, with a plain `streamText` call.
 *
 * An `{ type: 'error' }` part does not stop the source stream, so use
 * `controller.error()`.
 *
 * Offline: all models are mocks, no API key needed.
 *
 * Run:
 *   pnpm build && pnpm tsx examples/middleware-refusal-recovery.ts
 *
 * Output:
 *   retry with anthropic/claude-sonnet-4-5
 *   text: Chlorine sanitizes pool water.
 */
import {
  type LanguageModelMiddleware,
  streamText,
  wrapLanguageModel,
} from 'ai';
import { createRetryableModel, error } from 'ai-retry/language-model';
import { MockLanguageModel } from 'ai-test-kit/language';
import type { LanguageModelV4StreamPart } from '@ai-sdk/provider';

/** Errored into the stream when the buffered text matches a refusal phrase. */
class RefusalError extends Error {
  readonly phrase: string;
  constructor(phrase: string, bufferedText: string) {
    super(`Stream produced a refusal: ${JSON.stringify(bufferedText)}`);
    this.name = 'RefusalError';
    this.phrase = phrase;
  }
}

/** Lowercase the text and collapse whitespace, so a match works across delta boundaries. */
const normalize = (text: string) =>
  text.toLowerCase().replace(/\s+/g, ' ').trim();

/**
 * The stream parts that can come before the first text. Until the middleware
 * decides, it holds them back with the text deltas. Any other part is non-text
 * content, which is never a refusal, so the middleware forwards it at once. A
 * tool-call stream must not wait until the end.
 */
const PREAMBLE = new Set<LanguageModelV4StreamPart['type']>([
  'stream-start',
  'response-metadata',
  'text-start',
]);

/**
 * A `wrapLanguageModel` middleware that holds back the first parts until the
 * text shows if it is a refusal. On a match, it errors the stream with a
 * `RefusalError` before any part reaches the consumer. While the held text is
 * the start of a phrase, it continues to hold. When the text is different from
 * all phrases, or non-text content arrives, it sends all held parts and
 * forwards the rest without change.
 */
const refusalMiddleware = (
  phrases: ReadonlyArray<string>,
): LanguageModelMiddleware => {
  const targets = phrases.map(normalize);

  return {
    wrapStream: async ({ doStream }) => {
      const { stream, ...rest } = await doStream();

      let committed = false;
      let bufferedText = '';
      let bufferedParts: Array<LanguageModelV4StreamPart> = [];

      const transformed = stream.pipeThrough(
        new TransformStream<
          LanguageModelV4StreamPart,
          LanguageModelV4StreamPart
        >({
          transform(part, controller) {
            if (committed) return controller.enqueue(part);

            /**
             * Hold each part back until the text shows if it is a refusal. Thus
             * on a match, no part (also no stream-start, text-start or
             * metadata) reaches the consumer before the error.
             */
            bufferedParts.push(part);

            if (part.type === 'text-delta') {
              bufferedText += part.delta;
              const text = normalize(bufferedText);

              const matched = targets.find((phrase) => text.startsWith(phrase));
              if (matched) {
                /** Stop the source stream, and send an error that a condition can match. */
                controller.error(new RefusalError(matched, bufferedText));
                return;
              }
              /** The text is still the start of a phrase, so continue to hold. */
              if (targets.some((phrase) => phrase.startsWith(text))) return;
            } else if (PREAMBLE.has(part.type)) {
              /** A part that comes before the first text, so continue to hold. */
              return;
            }

            /**
             * The text is different from all phrases, or non-text content
             * arrived. This is a real answer, so send the held parts and forward
             * the rest.
             */
            committed = true;
            this.flush?.(controller);
          },
          /** Send the held parts, after the decision above and at the end of the stream. */
          flush(controller) {
            for (const part of bufferedParts) controller.enqueue(part);
            bufferedParts = [];
          },
        }),
      );

      return { stream: transformed, ...rest };
    },
  };
};

/** Refusals that a fallback model can answer. */
const REFUSALS = ["I'm sorry, but I cannot assist with that request."];

const baseModel = MockLanguageModel.from(
  "I'm sorry, but I cannot assist with that request.",
  { provider: 'openai', modelId: 'gpt-4o' },
);
const fallbackModel = MockLanguageModel.from('Chlorine sanitizes pool water.', {
  provider: 'anthropic',
  modelId: 'claude-sonnet-4-5',
});

/** Wrap the base model with the refusal middleware. */
const wrappedModel = wrapLanguageModel({
  model: baseModel,
  middleware: refusalMiddleware(REFUSALS),
});

/** Make it retryable, switching to the fallback on a `RefusalError`. */
const retryableModel = createRetryableModel({
  model: wrappedModel,
  retries: [error.isInstance(RefusalError).switch({ model: fallbackModel })],
  onRetry: ({ current }) => {
    console.log(
      `retry with ${current.model.provider}/${current.model.modelId}`,
    );
  },
});

const result = streamText({
  model: retryableModel,
  prompt: 'What sanitizes pool water?',
  maxRetries: 0,
});

console.log(`text: ${await result.text}`);
