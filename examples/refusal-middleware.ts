import {
  type LanguageModelMiddleware,
  streamText,
  wrapLanguageModel,
} from 'ai';
import { createRetryableModel, error } from 'ai-retry/language-model';
import { Language, MockLanguageModel } from 'ai-test-kit/language';

/** Canned refusals a content filter streams back instead of the model's answer. */
const REFUSALS = ["I'm sorry, but I can't assist with that request."];

/** Thrown once a stream turns out to be a refusal rather than an answer. */
class RefusalError extends Error {
  readonly refusal: string;

  constructor(refusal: string) {
    super(`Refusal detected: ${refusal}`);
    this.name = 'RefusalError';
    this.refusal = refusal;
  }
}

/**
 * Holds text deltas back while they still spell out the beginning of a known
 * refusal and throws as soon as one matches in full. The moment the buffer
 * diverges from every candidate, it is released and the rest of the stream is
 * forwarded untouched, so a normal answer is delayed by a few deltas at most.
 */
const detectRefusal = (): LanguageModelMiddleware => ({
  wrapStream: async ({ doStream }) => {
    const { stream, ...rest } = await doStream();

    let buffer = '';
    let id = '';
    let buffering = true;

    return {
      ...rest,
      stream: stream.pipeThrough(
        new TransformStream({
          transform(chunk, controller) {
            if (buffering && chunk.type === 'text-delta') {
              buffer += chunk.delta;
              id = chunk.id;

              /** A full refusal, and nothing has been emitted yet: bail out. */
              const refusal = REFUSALS.find((r) => buffer.startsWith(r));
              if (refusal) throw new RefusalError(refusal);

              /** Still the beginning of a refusal: keep holding it back. */
              if (REFUSALS.some((r) => r.startsWith(buffer))) return;

              /** Not a refusal: release the held-back text and forward the rest as it arrives. */
              buffering = false;
              controller.enqueue({ ...chunk, delta: buffer });
              buffer = '';
              return;
            }

            if (buffer) {
              controller.enqueue({ type: 'text-delta', id, delta: buffer });
              buffer = '';
            }

            controller.enqueue(chunk);
          },
          flush(controller) {
            if (buffer)
              controller.enqueue({ type: 'text-delta', id, delta: buffer });
          },
        }),
      ),
    };
  },
});

/** Streams the words of `text` as separate deltas, finishing with `finishReason: 'stop'`. */
const streamWords = (text: string) => [
  Language.streamStart(),
  ...Language.streamText(text, { separator: ' ' }),
  Language.streamFinish(),
];

/** The content filter streams a refusal, and the response still finishes with `stop`. */
const filtered = MockLanguageModel.from(
  { doStream: streamWords("I'm sorry, but I can't assist with that request.") },
  { provider: 'azure', modelId: 'gpt-4o' },
);

/** The same prompt on another provider answers normally. */
const unfiltered = MockLanguageModel.from(
  { doStream: streamWords('Sure! Here is the answer you asked for.') },
  { provider: 'openai', modelId: 'gpt-4o' },
);

const model = createRetryableModel({
  model: wrapLanguageModel({ model: filtered, middleware: detectRefusal() }),
  retries: [
    error.isInstance(RefusalError).switch({
      model: wrapLanguageModel({
        model: unfiltered,
        middleware: detectRefusal(),
      }),
    }),
  ],
});

const result = streamText({ model, prompt: 'Tell me about the movie xXx.' });

/** The consumer sees a single stream: no refusal, no error, just the fallback's answer. */
for await (const chunk of result.textStream) process.stdout.write(chunk);
