/**
 * Behavior matrix: how each retry wiring handles each mid-stream failure.
 *
 * Deterministic and offline (no API key): the real `@ai-sdk/openai` provider
 * runs over a synthetic `fetch` whose FIRST call fails in a controlled way and
 * whose later calls stream a clean "Hello, world!". A clean fallback model is
 * always available, so any non-recovery is the wiring's own limitation.
 *
 * Wirings (columns):
 *   - model : createRetryableModel alone, used as the streamText model (retry
 *             lives BELOW streamText, wrapping doStream).
 *   - stream: createRetryableStream alone (retry lives ABOVE streamText, re-runs
 *             the whole call).
 *   - both  : createRetryableStream wrapping createRetryableModel (composed).
 *
 * Cases (rows):
 *   - totalMs / stepMs / chunkMs : a streamText `timeout` deadline; attempt 1
 *     connects then stalls before content.
 *   - abortSignal : an external AbortSignal.timeout on the call; attempt 1 stalls.
 *   - error-before-content : attempt 1 emits preamble then the stream errors.
 *   - error-after-content  : attempt 1 emits one content chunk then errors.
 *
 * Outcome legend:
 *   RECOVER = full "Hello, world!" (failed over to the clean model)
 *   TRUNC   = partial text then stopped (committed, no fail-over)
 *   EMPTY   = no text, no throw (aborted before content, not recovered)
 *   THREW:x = the consumer saw error x
 *   HANG    = never settled within the guard window
 *   (n)     = number of HTTP calls made
 *
 * Run:
 *   pnpm build && pnpm tsx examples/retryable-stream-matrix.ts
 */
import { createOpenAI, type OpenAIProvider } from '@ai-sdk/openai';
import { streamText } from 'ai';
import { createRetryable } from 'ai-retry';
import {
  createRetryableStream,
  type RetryableStreamOptions,
} from 'ai-retry/experimental/stream';

/** Abandoned attempts (a hung streamText we stopped awaiting) reject later. */
process.on('unhandledRejection', () => {});

const enc = new TextEncoder();
const sse = (delta: Record<string, unknown>, finish: string | null = null) =>
  enc.encode(
    `data: ${JSON.stringify({ id: 's', object: 'chat.completion.chunk', model: 'm', choices: [{ index: 0, delta, finish_reason: finish }] })}\n\n`,
  );

const cleanBody = () =>
  new ReadableStream<Uint8Array>({
    start(c) {
      c.enqueue(sse({ role: 'assistant' }));
      c.enqueue(sse({ content: 'Hello, world!' }));
      c.enqueue(sse({}, 'stop'));
      c.enqueue(enc.encode('data: [DONE]\n\n'));
      c.close();
    },
  });

type FailKind = 'stall' | 'error-before' | 'error-after';

/** How attempt 1 misbehaves; cooperates with abort like a real socket. */
const failBody = (kind: FailKind) => (signal?: AbortSignal) =>
  new ReadableStream<Uint8Array>({
    async start(c) {
      c.enqueue(sse({ role: 'assistant' })); // preamble (no content yet)
      if (kind === 'error-after') c.enqueue(sse({ content: 'Partial ' }));
      if (kind === 'error-before' || kind === 'error-after') {
        /**
         * A real gap before the reset: without it the SDK drops the queued
         * content and the error looks pre-content. The gap lets the content
         * commit first, so `error-after` truly fires after the commit point.
         */
        await new Promise((r) => setTimeout(r, 25));
        c.error(new Error('mid-stream connection reset'));
        return;
      }
      // 'stall': stay idle until the deadline aborts us.
      signal?.addEventListener('abort', () => {
        try {
          c.error(signal.reason);
        } catch {}
      });
    },
  });

/** First HTTP call fails per `kind`; later calls stream cleanly. */
const counterProvider = (kind: FailKind) => {
  const calls = { count: 0 };
  const fetchImpl: typeof fetch = async (_i, init) => {
    calls.count += 1;
    const body =
      calls.count === 1
        ? failBody(kind)(init?.signal ?? undefined)
        : cleanBody();
    return new Response(body, {
      status: 200,
      headers: { 'content-type': 'text/event-stream' },
    });
  };
  return { openai: createOpenAI({ apiKey: 'x', fetch: fetchImpl }), calls };
};

const glue =
  (provider: OpenAIProvider, options: RetryableStreamOptions) =>
  (args: Omit<Parameters<typeof streamText>[0], 'model'>) => {
    const run = createRetryableStream(options);
    return run(
      (attempt) => {
        const { prompt: _p, ...overrides } = attempt.options;
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

type CaseDef = {
  key: string;
  kind: FailKind;
  /** Fresh per run: streamText timeout and/or an external abortSignal. */
  opts: () => Record<string, unknown>;
  /** abortSignal deadlines need a fresh per-attempt deadline on the fallback. */
  freshFallback?: number;
};

const DEADLINE = 200;
const cases: Array<CaseDef> = [
  {
    key: 'totalMs',
    kind: 'stall',
    opts: () => ({ timeout: { totalMs: DEADLINE } }),
  },
  {
    key: 'stepMs',
    kind: 'stall',
    opts: () => ({ timeout: { stepMs: DEADLINE } }),
  },
  {
    key: 'chunkMs',
    kind: 'stall',
    opts: () => ({ timeout: { chunkMs: DEADLINE } }),
  },
  {
    key: 'abortSignal',
    kind: 'stall',
    opts: () => ({ abortSignal: AbortSignal.timeout(DEADLINE) }),
    freshFallback: 5_000,
  },
  { key: 'error-before-content', kind: 'error-before', opts: () => ({}) },
  { key: 'error-after-content', kind: 'error-after', opts: () => ({}) },
];

const GUARD = 1_500;
const EXPECTED = 'Hello, world!';

/** Drive one cell to an outcome, never hanging the process. */
const runCell = async (
  build: () =>
    | Promise<{ textStream: AsyncIterable<string> }>
    | { textStream: AsyncIterable<string> },
  calls: { count: number },
): Promise<string> => {
  const work = (async () => {
    const result = await build();
    let text = '';
    let threw = '';
    try {
      for await (const delta of result.textStream) text += delta;
    } catch (e) {
      threw = (e as Error).name;
    }
    if (text === EXPECTED) return 'RECOVER';
    /** Committed content plus a re-streamed retry = duplicated / garbled. */
    if (text.includes(EXPECTED)) return 'DUP';
    if (text) return threw ? 'TRUNC+THREW' : 'TRUNC';
    if (threw) return `THREW:${threw}`;
    return 'EMPTY';
  })();
  const outcome = await Promise.race([
    work.catch((e) => `THREW:${(e as Error).name}`),
    new Promise<string>((r) => setTimeout(() => r('HANG'), GUARD)),
  ]);
  return `${outcome} (${calls.count})`;
};

const modelCell = (c: CaseDef) => {
  const { openai, calls } = counterProvider(c.kind);
  const model = createRetryable({
    model: openai.chat('a'),
    retries: [openai.chat('b')],
  });
  return runCell(
    () =>
      streamText({
        model,
        prompt: 'hi',
        maxRetries: 0,
        ...c.opts(),
        onError: () => {},
      }),
    calls,
  );
};

const streamCell = (c: CaseDef) => {
  const { openai, calls } = counterProvider(c.kind);
  const fallback = c.freshFallback
    ? { model: openai.chat('b'), timeout: c.freshFallback }
    : openai.chat('b');
  const run = glue(openai, { model: openai.chat('a'), retries: [fallback] });
  return runCell(() => run({ prompt: 'hi', ...c.opts() }), calls);
};

const bothCell = (c: CaseDef) => {
  const { openai, calls } = counterProvider(c.kind);
  const inner = createRetryable({
    model: openai.chat('a'),
    retries: [openai.chat('b')],
  });
  const fallback = c.freshFallback
    ? { model: openai.chat('c'), timeout: c.freshFallback }
    : openai.chat('c');
  const run = glue(openai, { model: inner, retries: [fallback] });
  return runCell(() => run({ prompt: 'hi', ...c.opts() }), calls);
};

const pad = (s: string, n: number) => s.padEnd(n);
console.log(
  pad('case', 22) + pad('model', 16) + pad('stream', 16) + 'model+stream',
);
console.log('-'.repeat(66));
for (const c of cases) {
  /** Sequential so the shared wall clock and call counters stay clean. */
  const model = await modelCell(c);
  const stream = await streamCell(c);
  const both = await bothCell(c);
  console.log(pad(c.key, 22) + pad(model, 16) + pad(stream, 16) + both);
}
console.log(
  [
    '',
    'RECOVER = full answer (failed over)      (n) = HTTP calls made',
    'TRUNC   = partial answer, no retry       EMPTY = aborted pre-content, no retry',
    'DUP     = committed content + re-stream (DUPLICATED / garbled output)',
    'HANG    = never settled within the guard window',
  ].join('\n'),
);
process.exit(0);
