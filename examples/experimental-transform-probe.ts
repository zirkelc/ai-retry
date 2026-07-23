/**
 * Probe: can `streamText`'s `experimental_transform` buffer text and stop the
 * stream like the refusal gate — and if it throws / emits an error / aborts,
 * what does the stream look like, and is it recoverable BELOW streamText (the
 * model layer, `createRetryable` wrapping `doStream`)?
 *
 * Pipeline fact (ai@7 source): user transforms are piped ABOVE the model, over
 * already-stitched `TextStreamPart`s, after `doStream` has resolved. So the
 * model layer should be blind to anything a transform does. This verifies that
 * and shows the emitted parts per strategy.
 *
 * Run:
 *   pnpm tsx examples/experimental-transform-probe.ts
 */
import { type StreamTextTransform, streamText } from 'ai';
import { createRetryable } from 'ai-retry';
import { createRetryableStream } from 'ai-retry/experimental/stream';
import { Language, MockLanguageModel } from 'ai-test-kit/language';

const REFUSAL = "i'm sorry, but i cannot assist";
const norm = (s: string) => s.toLowerCase().replace(/\s+/g, ' ').trim();

/** A model that streams a canned refusal split across deltas, finishing `stop`. */
const refuserChunks = [
  Language.streamStart(),
  ...Language.streamText(
    ["I'm sorry, ", 'but I cannot assist', ' with that.'],
    {
      id: '1',
    },
  ),
  Language.streamFinish(),
];
/** A clean fallback answer. */
const cleanChunks = [
  Language.streamStart(),
  ...Language.streamText(['Here ', 'is ', 'the answer.'], { id: '1' }),
  Language.streamFinish(),
];

/**
 * The action a transform takes once the buffered text matches the refusal.
 * `controller` is the transform's output controller; `stop` is `stopStream`.
 */
type OnMatch = (
  controller: TransformStreamDefaultController<any>,
  stop: () => void,
  buffered: string,
) => void;

/**
 * A refusal-detecting transform that buffers leading `text-delta`s (holding them
 * back) until it can tell a refusal from a real answer — the same shape as
 * a refusal transform, but at the streamText level. On a match it runs
 * `onMatch`; on divergence it flushes the held deltas and commits.
 */
const bufferingTransform =
  (onMatch: OnMatch): StreamTextTransform<any> =>
  ({ stopStream }) => {
    let buffer = '';
    let decided = false;
    const held: Array<any> = [];
    return new TransformStream({
      transform(chunk, controller) {
        if (decided) return controller.enqueue(chunk);
        if (chunk.type !== 'text-delta') return controller.enqueue(chunk);

        buffer += chunk.text;
        held.push(chunk);
        const text = norm(buffer);

        if (text.startsWith(REFUSAL)) {
          decided = true;
          held.length = 0; // drop the refusal text; emit only the action
          return onMatch(controller, stopStream, buffer);
        }
        if (REFUSAL.startsWith(text)) return; // still a prefix: keep holding
        decided = true; // diverged: a real answer
        for (const h of held) controller.enqueue(h);
        held.length = 0;
      },
      flush(controller) {
        for (const h of held) controller.enqueue(h);
      },
    });
  };

/** A minimal finish-step + finish pair, as the docs say stopStream requires. */
const emitFinish = (controller: TransformStreamDefaultController<any>) => {
  const usage = {
    inputTokens: undefined,
    outputTokens: undefined,
    totalTokens: undefined,
  };
  controller.enqueue({
    type: 'finish-step',
    response: {},
    usage,
    warnings: [],
  });
  controller.enqueue({
    type: 'finish',
    finishReason: 'stop',
    totalUsage: usage,
  });
};

const STRATEGIES: Record<string, OnMatch> = {
  'throw in transform': () => {
    throw new Error('refusal detected');
  },
  'enqueue error part': (c) => {
    c.enqueue({ type: 'error', error: new Error('refusal detected') });
  },
  'error part + stopStream': (c, stop) => {
    c.enqueue({ type: 'error', error: new Error('refusal detected') });
    stop();
  },
  'stopStream only': (_c, stop) => {
    stop();
  },
  'stopStream + finish': (c, stop) => {
    stop();
    emitFinish(c);
  },
  'enqueue abort part': (c) => {
    c.enqueue({ type: 'abort', reason: 'refusal detected' });
  },
};

/** Iterate `result.stream`, bounded so a stalled/terminated stream can't hang. */
const collect = async (result: { stream: AsyncIterable<any> }) => {
  const parts: Array<any> = [];
  let threw: string | undefined;
  const drain = (async () => {
    for await (const part of result.stream) parts.push(part);
  })();
  await Promise.race([
    drain.catch((e) => {
      threw = `${(e as Error).name}: ${(e as Error).message}`;
    }),
    new Promise((r) => setTimeout(r, 1_500)),
  ]);
  return { parts, threw };
};

const summarize = (parts: Array<any>) => {
  const types = parts.map((p) => p.type).join(',');
  const errorPart = parts.find((p) => p.type === 'error');
  const abortPart = parts.find((p) => p.type === 'abort');
  const text = parts
    .filter((p) => p.type === 'text-delta')
    .map((p) => p.text)
    .join('');
  return { types, errorPart, abortPart, text };
};

const run = async (label: string, onMatch: OnMatch) => {
  console.log(`\n=== ${label} ===`);

  // 1. Plain streamText: what does the transformed stream look like?
  const onErrors: Array<unknown> = [];
  const plain = streamText({
    model: MockLanguageModel.from({ doStream: refuserChunks }),
    prompt: 'hi',
    experimental_transform: bufferingTransform(onMatch),
    onError: ({ error }) => {
      onErrors.push(error);
    },
  });
  const { parts, threw } = await collect(plain);
  const s = summarize(parts);
  console.log(`  parts:     [${s.types}]`);
  console.log(`  text:      ${JSON.stringify(s.text)}`);
  console.log(`  stream threw: ${threw ?? '(no — ended/terminated cleanly)'}`);
  if (s.errorPart)
    console.log(`  error part: ${(s.errorPart.error as Error)?.message}`);
  if (s.abortPart) console.log(`  abort part: ${JSON.stringify(s.abortPart)}`);
  console.log(
    `  onError fired: ${onErrors.length}${onErrors.length ? ` (${(onErrors[0] as Error)?.message})` : ''}`,
  );

  // 2. Model layer: does createRetryable (wrapping doStream) recover it?
  const refuser = MockLanguageModel.from({ doStream: refuserChunks });
  const modelFallback = MockLanguageModel.from({ doStream: cleanChunks });
  const wrapped = streamText({
    model: createRetryable({ model: refuser, retries: [modelFallback] }),
    prompt: 'hi',
    maxRetries: 0,
    experimental_transform: bufferingTransform(onMatch),
    onError: () => {},
  });
  await collect(wrapped);
  console.log(
    `  model-layer recovery: fallback.doStream = ${modelFallback.doStream.mock.calls.length} -> ${modelFallback.doStream.mock.calls.length ? 'RECOVERED' : 'not recoverable'}`,
  );

  // 3. Call layer: does createRetryableStream (reading result.stream) fail over?
  const primary = MockLanguageModel.from({ doStream: refuserChunks });
  const callFallback = MockLanguageModel.from({ doStream: cleanChunks });
  const run = createRetryableStream({
    model: primary,
    retries: [callFallback],
  });
  let callText = '';
  try {
    const result = await run((attempt) =>
      streamText({
        model: attempt.model,
        prompt: 'hi',
        abortSignal: attempt.abortSignal,
        experimental_transform: bufferingTransform(onMatch),
        onError: () => {},
      }),
    );
    for await (const part of result.stream)
      if (part.type === 'text-delta') callText += part.text;
  } catch (e) {
    callText = `<threw: ${(e as Error).message}>`;
  }
  console.log(
    `  call-layer recovery: fallback.doStream = ${callFallback.doStream.mock.calls.length}, text=${JSON.stringify(callText)} -> ${callFallback.doStream.mock.calls.length ? 'RECOVERED' : 'not recoverable'}`,
  );
};

// Sanity: a real answer that opens like a refusal must pass through intact.
const passthrough = await (async () => {
  const result = streamText({
    model: MockLanguageModel.from({
      doStream: [
        Language.streamStart(),
        ...Language.streamText(["I'm sorry ", 'to hear that. Here is help.'], {
          id: '1',
        }),
        Language.streamFinish(),
      ],
    }),
    prompt: 'hi',
    experimental_transform: bufferingTransform(() => {
      throw new Error('should not fire');
    }),
    onError: () => {},
  });
  return summarize((await collect(result)).parts).text;
})();
console.log('=== baseline: divergent answer passes through ===');
console.log(`  text: ${JSON.stringify(passthrough)}`);

for (const [label, onMatch] of Object.entries(STRATEGIES)) {
  await run(label, onMatch);
}

process.exit(0);
