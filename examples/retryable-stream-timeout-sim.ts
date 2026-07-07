/**
 * Deterministic (no API key) demonstration of how stream *timeouts* interact
 * with `createRetryableStream` fail-over — the case that most often surprises
 * people in production. It uses the real `@ai-sdk/openai` provider over a
 * synthetic, controllable "network".
 *
 * The one rule, again: a stream can only fail over BEFORE its first content
 * part. Timeouts split cleanly along that line:
 *
 *   A. A "time to first byte" read-timeout fires BEFORE any content -> the
 *      attempt has not committed, so it fails over and the next model answers.
 *      This is the timeout you usually WANT for a stalled connection.
 *
 *   B. `streamText`'s built-in `timeout: { chunkMs }` / `{ stepMs }` only starts
 *      ticking once content is already flowing, so when it fires it is AFTER the
 *      commit point. The stream cannot fail over; the abort just ends the
 *      stream and the caller is left with a TRUNCATED answer and no retry.
 *      If your deployment "randomly returned half a response with no error",
 *      this is almost certainly why.
 *
 * Takeaway: to recover a stalled stream, put the deadline on the connection /
 * first byte (scenario A), not on `chunkMs` / `stepMs` mid-stream (scenario B).
 *
 * Run:
 *   pnpm build && pnpm tsx examples/retryable-stream-timeout-sim.ts
 */
import { createOpenAI } from '@ai-sdk/openai';
import { streamText } from 'ai';
import {
  createRetryableStream,
  type RetryableStreamOptions,
} from 'ai-retry/experimental/stream';

const encoder = new TextEncoder();
const chunk = (delta: Record<string, unknown>, finish: string | null = null) =>
  encoder.encode(
    `data: ${JSON.stringify({ id: 'sim', object: 'chat.completion.chunk', model: 'gpt-4o', choices: [{ index: 0, delta, finish_reason: finish }] })}\n\n`,
  );

const sseResponse = (body: ReadableStream<Uint8Array>) =>
  new Response(body, {
    status: 200,
    headers: { 'content-type': 'text/event-stream' },
  });

/** A complete, well-formed chat-completion stream. */
const cleanBody = () =>
  new ReadableStream<Uint8Array>({
    start(controller) {
      controller.enqueue(chunk({ role: 'assistant' }));
      controller.enqueue(chunk({ content: 'Exponential backoff ' }));
      controller.enqueue(chunk({ content: 'spaces out retries.' }));
      controller.enqueue(chunk({}, 'stop'));
      controller.enqueue(encoder.encode('data: [DONE]\n\n'));
      controller.close();
    },
  });

/** Connects (HTTP 200) but never sends a byte — a silent, stalled upstream. */
const silentBody = (signal?: AbortSignal) =>
  new ReadableStream<Uint8Array>({
    start(controller) {
      signal?.addEventListener('abort', () => {
        try {
          controller.error(signal.reason);
        } catch {}
      });
    },
  });

/** Streams one content delta, then goes idle forever. */
const contentThenIdleBody = (signal?: AbortSignal) =>
  new ReadableStream<Uint8Array>({
    start(controller) {
      controller.enqueue(chunk({ role: 'assistant' }));
      controller.enqueue(chunk({ content: 'Exponential backoff ' }));
      signal?.addEventListener('abort', () => {
        try {
          controller.error(signal.reason);
        } catch {}
      });
    },
  });

/**
 * A reusable `fetch` wrapper that enforces a "time to first byte" deadline: if
 * the upstream sends no data within `ms`, the call rejects with a TimeoutError.
 * This is the piece that makes a stalled connection RECOVERABLE, because it
 * fails before any content commits. Copy this into your app.
 */
const withReadTimeout =
  (baseFetch: typeof fetch, ms: number): typeof fetch =>
  async (input, init) => {
    const response = await baseFetch(input, init);
    if (!response.body) return response;
    const reader = response.body.getReader();

    const first = await new Promise<
      Awaited<ReturnType<ReadableStreamDefaultReader<Uint8Array>['read']>>
    >((resolve, reject) => {
      const timer = setTimeout(() => {
        void reader.cancel();
        reject(
          Object.assign(new Error(`no first byte within ${ms}ms`), {
            name: 'TimeoutError',
          }),
        );
      }, ms);
      reader.read().then(
        (result) => {
          clearTimeout(timer);
          resolve(result);
        },
        (error) => {
          clearTimeout(timer);
          reject(error);
        },
      );
    });

    /** First byte arrived in time: re-emit it, then forward the rest. */
    const body = new ReadableStream<Uint8Array>({
      start(controller) {
        if (!first.done && first.value) controller.enqueue(first.value);
        if (first.done) controller.close();
      },
      async pull(controller) {
        const { done, value } = await reader.read();
        if (done) return controller.close();
        controller.enqueue(value);
      },
      cancel(reason) {
        void reader.cancel(reason);
      },
    });
    return new Response(body, {
      status: response.status,
      statusText: response.statusText,
      headers: response.headers,
    });
  };

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
        onError: () => {},
      } as Parameters<typeof streamText>[0]);
    },
    { abortSignal: args.abortSignal },
  );
};

const EXPECTED = 'Exponential backoff spaces out retries.';

/**
 * Scenario A — a read-timeout guards the connection. Attempt 1's upstream is
 * silent; the wrapper rejects at 100ms (before content) and fail-over kicks in.
 */
const scenarioA = async () => {
  console.log(
    '\n=== A. read-timeout before first byte (expect: fail-over) ===',
  );
  let attempt = 0;
  const raw: typeof fetch = async (_input, init) => {
    attempt += 1;
    console.log(
      `   fetch #${attempt}: ${attempt === 1 ? 'silent upstream' : 'clean'}`,
    );
    return sseResponse(
      attempt === 1 ? silentBody(init?.signal ?? undefined) : cleanBody(),
    );
  };
  const openai = createOpenAI({
    apiKey: 'sim',
    fetch: withReadTimeout(raw, 100),
  });

  const result = await retryableStreamText(
    { model: openai.chat('gpt-4o-mini'), retries: [openai.chat('gpt-4o')] },
    { prompt: 'What is exponential backoff?' },
  );

  let text = '';
  for await (const delta of result.textStream) text += delta;
  console.log(`   attempts: ${attempt}, text: "${text}"`);
  console.log(
    `   -> ${text === EXPECTED ? 'RECOVERED (full answer)' : 'INCOMPLETE'}`,
  );
};

/**
 * Scenario B — `streamText`'s chunkMs deadline. Attempt 1 streams one delta then
 * goes idle; chunkMs fires ~100ms later, but content has already committed, so
 * there is no fail-over. The caller gets a truncated answer, no retry, no throw.
 */
const scenarioB = async () => {
  console.log(
    '\n=== B. chunkMs idle-timeout after content (expect: no retry) ===',
  );
  let attempt = 0;
  const raw: typeof fetch = async (_input, init) => {
    attempt += 1;
    console.log(
      `   fetch #${attempt}: ${attempt === 1 ? 'content then idle' : 'clean'}`,
    );
    return sseResponse(
      attempt === 1
        ? contentThenIdleBody(init?.signal ?? undefined)
        : cleanBody(),
    );
  };
  const openai = createOpenAI({ apiKey: 'sim', fetch: raw });

  const result = await retryableStreamText(
    { model: openai.chat('gpt-4o-mini'), retries: [openai.chat('gpt-4o')] },
    { prompt: 'What is exponential backoff?', timeout: { chunkMs: 100 } },
  );

  let text = '';
  try {
    for await (const delta of result.textStream) text += delta;
  } catch (error) {
    console.log(`   textStream threw: ${(error as Error).message}`);
  }
  console.log(`   attempts: ${attempt}, text: "${text}"`);
  console.log(
    `   -> ${text === EXPECTED ? 'RECOVERED' : 'TRUNCATED (committed on first delta, no fail-over)'}`,
  );
};

await scenarioA();
await scenarioB();
console.log(
  '\nTakeaway: guard the first byte (recoverable) — chunkMs/stepMs fire after\ncontent commits, so they truncate instead of failing over.',
);
process.exit(0);
