/**
 * Recover a canned refusal at the MODEL layer using a standard
 * `wrapLanguageModel` middleware — no ai-retry-specific option required.
 *
 * A natural-language refusal ("I'm sorry, but I cannot assist…") streams as
 * ordinary `text-delta` parts finishing `stop` — no error, no `content-filter`
 * finish reason — so nothing above the model can see it as a failure. The
 * middleware's `wrapStream` wraps `doStream`, so it sits AT the model boundary:
 * it holds back every leading part (stream-start, text-start, metadata, the text
 * deltas), and on matching a refusal it calls `controller.error(new RefusalError(
 * ...))` before anything has reached the consumer.
 *
 * `controller.error` does two things at once:
 *   1. `pipeThrough` cancels the source on a transform error, so the model
 *      stream is torn down (the upstream request is aborted — no wasted tokens).
 *   2. the consumer's read rejects with the `RefusalError`, which
 *      `createRetryableModel` catches BEFORE content and runs through its retry
 *      conditions — so `error.isInstance(RefusalError).switch(...)` fails over,
 *      under plain `streamText`, with no call-layer wrapper.
 *
 * (Enqueuing an `{ type: 'error' }` part instead would NOT stop the source — the
 * model keeps streaming — so `controller.error` is the right tool here.)
 *
 * Run:
 *   pnpm build && pnpm tsx examples/middleware-refusal-recovery.ts
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

/** Lowercase + collapse whitespace, so matching survives delta boundaries. */
const normalize = (text: string) =>
  text.toLowerCase().replace(/\s+/g, ' ').trim();

/**
 * The only stream parts that precede the leading text. While undecided we hold
 * these back with the text deltas; anything else means non-text content, which
 * is never a refusal, so it commits immediately (a tool-call stream must not be
 * buffered to the end).
 */
const PREAMBLE = new Set<LanguageModelV4StreamPart['type']>([
  'stream-start',
  'response-metadata',
  'text-start',
]);

/**
 * A `wrapLanguageModel` middleware that holds back every leading part until the
 * text lets it decide, and on matching a refusal phrase errors the stream with a
 * `RefusalError` (which stops the source and is recoverable by an error-based
 * retry) before any part has reached the consumer. While the buffered text is
 * still a prefix of some phrase it keeps holding; once it diverges (or non-text
 * content arrives) it flushes everything held and forwards the rest untouched.
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
             * Hold every part back until the leading text lets us decide, so a
             * matched refusal errors before ANYTHING reaches the consumer — no
             * stream-start, text-start or metadata leaked ahead of the error.
             */
            bufferedParts.push(part);

            if (part.type === 'text-delta') {
              bufferedText += part.delta;
              const text = normalize(bufferedText);

              const matched = targets.find((phrase) => text.startsWith(phrase));
              if (matched) {
                /** Stop the source AND deliver a matchable error. */
                controller.error(new RefusalError(matched, bufferedText));
                return;
              }
              /** Still an inconclusive prefix of a phrase: keep holding. */
              if (targets.some((phrase) => phrase.startsWith(text))) return;
            } else if (PREAMBLE.has(part.type)) {
              /** A part that precedes the leading text: keep holding. */
              return;
            }

            /**
             * The text diverged from every phrase, or non-text content arrived:
             * a real answer, not a refusal. Flush what's held, forward the rest.
             */
            committed = true;
            this.flush?.(controller);
          },
          /** Flush the held parts: on commit above, and at stream end. */
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

/** Known false-positive refusals worth failing over from. */
const REFUSALS = ["I'm sorry, but I cannot assist with that request."];

const baseModel = MockLanguageModel.from(
  "I'm sorry, but I cannot assist with that request.",
);
const fallbackModel = MockLanguageModel.from('OK');

/** Wrap the base model with the refusal middleware. */
const wrappedModel = wrapLanguageModel({
  model: baseModel,
  middleware: refusalMiddleware(REFUSALS),
});

/** Make it retryable, switching to the fallback on a `RefusalError`. */
const model = createRetryableModel({
  model: wrappedModel,
  retries: [error.isInstance(RefusalError).switch({ model: fallbackModel })],
});

const result = streamText({ model, prompt: 'What sanitizes pool water?' });
const text = await result.text;

console.log(`baseModel.doStream:     ${baseModel.doStream.mock.calls.length}`);
console.log(
  `fallbackModel.doStream: ${fallbackModel.doStream.mock.calls.length}`,
);
console.log(`text: ${JSON.stringify(text)}`);
process.exit(0);
