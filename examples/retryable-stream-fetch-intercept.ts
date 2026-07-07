/**
 * Runnable end-to-end test of `createRetryableStream` against a REAL provider
 * (OpenAI), with a custom `fetch` that severs the streaming response mid-flight
 * to simulate a network drop / load-balancer reset.
 *
 * It runs two scenarios that expose the one rule that governs whether a stream
 * can fail over:
 *
 *   A. The connection drops BEFORE any text has streamed  -> fail-over fires,
 *      the next model produces the answer, the caller never sees the error.
 *   B. The connection drops AFTER some text has streamed   -> the attempt has
 *      already "committed"; retrying would duplicate output, so the error is
 *      surfaced to the caller instead. NO retry happens. This is by design and
 *      is the most common surprise in production.
 *
 * The commit boundary is the first content part (the same point the AI SDK's
 * `onChunk` fires on). Everything before it (stream-start, response metadata,
 * step-start, reasoning-start, ...) is preamble and is safe to fail over.
 *
 * Run (needs a built dist + an OpenAI key):
 *   pnpm build
 *   OPENAI_API_KEY=sk-... pnpm tsx examples/retryable-stream-fetch-intercept.ts
 *   # or: pnpm tsx --env-file=.env examples/retryable-stream-fetch-intercept.ts
 */
import { createOpenAI } from '@ai-sdk/openai';
import { streamText } from 'ai';
import {
  createRetryableStream,
  type RetryableStreamOptions,
} from 'ai-retry/experimental/stream';

const apiKey = process.env.OPENAI_API_KEY;
if (!apiKey) {
  console.error('Set OPENAI_API_KEY to run this example.');
  process.exit(1);
}

type CutMode = 'before-content' | 'after-content';

/**
 * Re-emit an OpenAI SSE byte stream, then `error()` it to mimic a connection
 * that dies mid-response. `before-content` cuts the moment the first content
 * delta would be forwarded (so the consumer never sees content); `after-content`
 * lets a couple of content deltas through first (so the attempt commits).
 */
const cutSseStream = (
  source: ReadableStream<Uint8Array>,
  mode: CutMode,
): ReadableStream<Uint8Array> => {
  const reader = source.getReader();
  const decoder = new TextDecoder();
  const encoder = new TextEncoder();
  let buffer = '';
  let contentDeltas = 0;

  /** True when a complete SSE event carries a non-empty assistant content delta. */
  const isContentEvent = (event: string): boolean => {
    const dataLine = event.split('\n').find((line) => line.startsWith('data:'));
    const payload = dataLine?.slice('data:'.length).trim();
    if (!payload || payload === '[DONE]') return false;
    try {
      return Boolean(JSON.parse(payload)?.choices?.[0]?.delta?.content);
    } catch {
      return false;
    }
  };

  const reset = (where: string) =>
    new Error(`simulated mid-stream connection reset (${where})`);

  return new ReadableStream<Uint8Array>({
    async pull(controller) {
      const { done, value } = await reader.read();
      if (done) {
        controller.close();
        return;
      }

      buffer += decoder.decode(value, { stream: true });

      /** SSE events are delimited by a blank line; keep the trailing partial. */
      const events = buffer.split('\n\n');
      buffer = events.pop() ?? '';

      for (const event of events) {
        if (!event.trim()) continue;
        const hasContent = isContentEvent(event);

        if (mode === 'before-content' && hasContent) {
          /** Die before the first content delta reaches the consumer. */
          controller.error(reset('pre-content'));
          return;
        }

        controller.enqueue(encoder.encode(`${event}\n\n`));

        if (mode === 'after-content' && hasContent && ++contentDeltas >= 2) {
          /** Some text already streamed: the attempt has committed. */
          controller.error(reset('post-content'));
          return;
        }
      }
    },
    cancel(reason) {
      void reader.cancel(reason);
    },
  });
};

/**
 * A `fetch` that tampers with ONLY the first streaming call it sees (the first
 * retry attempt), passing every later call (the fail-over attempt) straight
 * through to the real network.
 */
const makeInterceptingFetch = (mode: CutMode): typeof fetch => {
  let calls = 0;
  return async (input, init) => {
    const attempt = ++calls;
    const response = await fetch(input, init);
    if (attempt > 1 || !response.body) {
      console.log(`   fetch #${attempt}: passthrough (untampered)`);
      return response;
    }
    console.log(`   fetch #${attempt}: tampering -> ${mode}`);
    return new Response(cutSseStream(response.body, mode), {
      status: response.status,
      statusText: response.statusText,
      headers: response.headers,
    });
  };
};

/**
 * Drop-in `streamText` glue on top of `createRetryableStream`: re-run the whole
 * `streamText` call per attempt with the attempt's model and fresh deadline
 * signal, deciding commit/fail-over from `fullStream`.
 */
const retryableStreamText = (
  options: RetryableStreamOptions,
  args: Omit<Parameters<typeof streamText>[0], 'model'>,
) => {
  const run = createRetryableStream(options);
  return run(
    (attempt) => {
      const { prompt: _prompt, ...overrides } = attempt.options;
      return streamText({
        ...args,
        ...overrides,
        model: attempt.model,
        abortSignal: attempt.abortSignal,
        /** This wrapper detects errors from `fullStream`; mute the SDK's logger. */
        onError: () => {},
      } as Parameters<typeof streamText>[0]);
    },
    { abortSignal: args.abortSignal },
  );
};

const runScenario = async (label: string, mode: CutMode) => {
  console.log(`\n=== ${label} ===`);

  /**
   * A fresh provider per scenario so each gets its own intercepting fetch and
   * call counter. Base attempt fails (tampered); the fallback model is a
   * different model key so it is allowed to retry on any error.
   */
  const openai = createOpenAI({ apiKey, fetch: makeInterceptingFetch(mode) });

  const result = await retryableStreamText(
    {
      /**
       * `openai.chat(...)` forces the Chat Completions SSE format this demo's
       * byte-level fetch interception understands. The library itself is format
       * agnostic (it works on normalized AI SDK stream parts), so the default
       * `openai(...)` Responses API works too — only this example's cut logic
       * is chat-shaped. The fallback is a different model key so it is allowed
       * to retry on the (any-error) fallback entry.
       */
      model: openai.chat('gpt-4o-mini'),
      retries: [openai.chat('gpt-4o')],
    },
    { prompt: 'In one short sentence, what is exponential backoff?' },
  );

  let text = '';
  try {
    for await (const delta of result.textStream) {
      text += delta;
      process.stdout.write(delta);
    }
    const { modelId } = await result.response;
    console.log(`\n   -> committed model: ${modelId}`);
    console.log(`   -> RECOVERED: full answer received (${text.length} chars)`);
  } catch (error) {
    console.log(
      `\n   -> NOT RECOVERED: error surfaced after ${text.length} chars of content`,
    );
    console.log(`   -> ${(error as Error).message}`);
  }
};

await runScenario(
  'A. drop BEFORE content  (expect: fail-over to gpt-4o, full answer)',
  'before-content',
);
await runScenario(
  'B. drop AFTER content   (expect: no retry, error surfaces to caller)',
  'after-content',
);

console.log(
  '\nTakeaway: a stream can only fail over before its first content part.',
);
