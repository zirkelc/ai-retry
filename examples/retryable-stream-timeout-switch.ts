/**
 * Switch models when a streamed call times out, using `createRetryableStream`.
 *
 * A deadline set with `streamText`'s own `timeout: { firstChunkMs | stepMs |
 * totalMs }` cannot be recovered by a model wrapped with `createRetryableModel`:
 * that retries BELOW `streamText`, but the deadline lives ON the call, and once
 * it fires `streamText` finalizes the stream as aborted and discards whatever a
 * lower retry produced (issue #50). `createRetryableStream` re-runs the WHOLE
 * call with the next model, so fail-over works.
 *
 * Any pre-content deadline works: `firstChunkMs`, `stepMs`, and `totalMs` all
 * start counting before the first content part, so a stall trips them while the
 * attempt can still fail over (only `chunkMs`/`toolMs` fire after content, too
 * late). This example uses `firstChunkMs` (the "time to first byte" case): when
 * attempt 1 stalls, the deadline surfaces as a `TimeoutError`, and
 * `timeout().switch({ model })` re-runs with the fallback, which answers.
 *
 * The stall is simulated (no API key needed): the slow model opens its stream
 * then hangs until its abort signal fires, so `firstChunkMs` always trips.
 *
 * Run:
 *   pnpm build && pnpm tsx examples/retryable-stream-timeout-switch.ts
 */
import { streamText } from 'ai';
import { createRetryableStream } from 'ai-retry/experimental/stream';
import { timeout } from 'ai-retry/language-model';
import { Language, MockLanguageModel } from 'ai-test-kit/language';
import type { LanguageModelV4CallOptions } from '@ai-sdk/provider';

/**
 * A model whose stream opens (`stream-start`) but never sends content: it errors
 * only when its abort signal fires. `streamText`'s `firstChunkMs` deadline is
 * what fires that signal, so the attempt fails before any content commits.
 */
const stallModel = () =>
  MockLanguageModel.from({
    doStream: async ({ abortSignal }: LanguageModelV4CallOptions) => ({
      stream: new ReadableStream({
        start(controller) {
          controller.enqueue(Language.streamStart());
          abortSignal?.addEventListener(
            'abort',
            () => controller.error(abortSignal.reason),
            { once: true },
          );
        },
      }),
    }),
  });

const stall = stallModel();
const fast = MockLanguageModel.from('Chlorine tablets sanitize pool water.');

const run = createRetryableStream({
  model: stall,
  retries: [
    /** On a pre-content timeout, fail over to the fast model. */
    timeout().switch({ model: fast }),
  ],
});

/**
 * Each attempt builds its own `streamText`, wiring in `attempt.model` and a
 * `firstChunkMs` deadline; `maxRetries: 0` leaves retrying to the wrapper.
 * `createRetryableStream` resolves once an attempt commits (first content part),
 * so the returned result is the winning stream — drive it as usual.
 */
const result = await run((attempt) =>
  streamText({
    model: attempt.model,
    abortSignal: attempt.abortSignal,
    prompt: 'What sanitizes pool water?',
    /** Any pre-content deadline works here: firstChunkMs, stepMs, or totalMs. */
    timeout: { firstChunkMs: 100 },
    maxRetries: 0,
    onError: () => {},
  }),
);

let text = '';
for await (const delta of result.textStream) text += delta;

console.log(`stall.doStream: ${stall.doStream.mock.calls.length}`);
console.log(`fast.doStream:  ${fast.doStream.mock.calls.length}`);
console.log(`text: ${JSON.stringify(text)}`);
process.exit(0);
