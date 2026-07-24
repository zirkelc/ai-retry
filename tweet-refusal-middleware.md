# Tweet: refusal messages in streaming

## Tweet text

I finally found a way to deal with refusal messages in a streaming context.

This issue was bugging me for sooo long.

Let me explain:

Azure has a very sensitive content filter. Even certain phrases like "xxx" trip the filter and make the model refuse with "I'm sorry, but I can't assist with that request.".

By now I'm certain they use some kind of fuzzy keyword matching, maybe even the same one they use for Bing safe search. It can't come from the model itself, because the same phrase on the same model on OpenAI works just fine.

In a non-streaming context it's not a problem: you get finish-reason=content-filter and you can just switch the provider or the model.

The problem is streaming. The endpoint actually streams back the refusal message and ends with finish-reason=stop. And once the stream has emitted content, you can't switch anymore, because the user has already seen the refusal.

The solution I'm using now is a language model middleware in combination with my ai-retry library.

The middleware buffers text chunks and compares the current buffer against a list of known refusal messages. Azure always sends the same message, so it's easy to detect.

It only buffers until the first chunk that diverges from the refusal message, so it doesn't add latency in the normal case.

Once it detects a refusal message, it throws a custom `RefusalError`.

ai-retry then matches that error with `error.isInstance(RefusalError)` and switches to a different model.

The user sees none of this: no refusal message and no error. The client side sees a single stream, because ai-retry swaps the model but keeps the same stream open.

## Code snippet (for ray.so)

```ts
import { type LanguageModelMiddleware, streamText, wrapLanguageModel } from 'ai';
import { createRetryableModel, error } from 'ai-retry/language-model';
import { Language, MockLanguageModel } from 'ai-test-kit/language';

const REFUSALS = ["I'm sorry, but I can't assist with that request."];

class RefusalError extends Error {
  override name = 'RefusalError';
}

/** Holds text back while it still spells out a known refusal,
 *  and throws once one matches in full. */
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

              /** Not a refusal: release the buffer and forward the rest as it arrives. */
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

const stream = (text: string) => ({
  doStream: [
    Language.streamStart(),
    ...Language.streamText(text, { separator: ' ' }),
    Language.streamFinish(),
  ],
});

/** Azure streams the refusal and still finishes with `stop`,
 *  while OpenAI answers the same prompt. */
const azure = MockLanguageModel.from(
  stream("I'm sorry, but I can't assist with that request."),
);
const openai = MockLanguageModel.from(
  stream('Sure! Here is the answer you asked for.'),
);

const model = createRetryableModel({
  model: wrapLanguageModel({ model: azure, middleware: detectRefusal() }),
  retries: [
    error.isInstance(RefusalError).switch({
      model: wrapLanguageModel({ model: openai, middleware: detectRefusal() }),
    }),
  ],
});

const result = streamText({ model, prompt: 'Tell me about the movie xXx.' });

/** One stream, no refusal, no error: the fallback's answer. */
for await (const chunk of result.textStream) process.stdout.write(chunk);
```

## Notes

- The snippet runs as-is (`pnpm tsx`) against `ai@7`, `ai-retry@2` and `ai-test-kit@3`. Output: `Sure! Here is the answer you asked for.`
- A longer, more heavily commented version of the same program is in `examples/refusal-middleware.ts`.
- Why the retry is still possible: the middleware never forwards the buffered deltas, so from ai-retry's point of view no content chunk has been emitted yet, and a fallback before the first content chunk is exactly what it supports.
- Verified paths: refusal (falls back), normal answer (streams straight through), and text that starts like a refusal but diverges (the buffer is released in order, nothing is lost).
