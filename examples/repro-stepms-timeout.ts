/**
 * Repro: what does `streamText`'s `timeout: { stepMs }` emit on `stream`,
 * and is it catchable via try/catch?
 *
 * Unlike `chunkMs` (which only starts ticking once content is flowing and
 * resets on every content chunk), `stepMs` starts ticking at the START of the
 * step — before the first byte. So a stall BEFORE any content trips stepMs
 * pre-commit, which is the window where fail-over is still possible.
 *
 * This probes three upstream behaviours against a small stepMs, and for each
 * prints the exact `stream` parts, whether iteration threw, and (when it
 * aborted) the abort part's serialized `reason`.
 *
 * Run:
 *   pnpm tsx examples/repro-stepms-timeout.ts
 */
import { createOpenAI } from '@ai-sdk/openai';
import { streamText } from 'ai';

const enc = new TextEncoder();
const sse = (delta: Record<string, unknown>, finish: string | null = null) =>
  enc.encode(
    `data: ${JSON.stringify({ id: 'x', object: 'chat.completion.chunk', model: 'gpt-4o', choices: [{ index: 0, delta, finish_reason: finish }] })}\n\n`,
  );

/** 200 OK, then never sends a byte. stepMs must fire before any content. */
const silent = (signal?: AbortSignal) =>
  new ReadableStream<Uint8Array>({
    start(c) {
      signal?.addEventListener('abort', () => {
        try {
          c.error(signal.reason);
        } catch {}
      });
    },
  });

/** Emits one content delta, then stalls. stepMs fires AFTER commit. */
const contentThenStall = (signal?: AbortSignal) =>
  new ReadableStream<Uint8Array>({
    start(c) {
      c.enqueue(sse({ role: 'assistant' }));
      c.enqueue(sse({ content: 'Hello ' }));
      signal?.addEventListener('abort', () => {
        try {
          c.error(signal.reason);
        } catch {}
      });
    },
  });

const makeModel = (body: (s?: AbortSignal) => ReadableStream<Uint8Array>) =>
  createOpenAI({
    apiKey: 'test',
    fetch: async (_i, init) =>
      new Response(body(init?.signal ?? undefined), {
        status: 200,
        headers: { 'content-type': 'text/event-stream' },
      }),
  }).chat('gpt-4o');

const run = async (
  label: string,
  body: (s?: AbortSignal) => ReadableStream<Uint8Array>,
) => {
  const result = streamText({
    model: makeModel(body),
    prompt: 'hi',
    timeout: { stepMs: 200 },
    onError: () => {},
  });

  const t0 = Date.now();
  const parts: Array<Record<string, unknown>> = [];
  let threw: string | undefined;
  try {
    for await (const p of result.stream) {
      parts.push(p as Record<string, unknown>);
    }
  } catch (e) {
    threw = `${(e as Error).name}: ${(e as Error).message}`;
  }
  const dt = Date.now() - t0;
  const types = parts.map((p) => p.type).join(',');
  const abortPart = parts.find((p) => p.type === 'abort');
  console.log(`\n### ${label} (@${dt}ms)`);
  console.log(`  parts:      [${types}]`);
  console.log(
    `  threw:      ${threw ?? '(no throw — iteration ended cleanly)'}`,
  );
  if (abortPart) {
    console.log(`  abort part: ${JSON.stringify(abortPart)}`);
  }
  const finish = parts.find((p) => p.type === 'finish') as
    | { finishReason?: string }
    | undefined;
  if (finish) console.log(`  finishReason: ${finish.finishReason}`);
};

await run('silent upstream (stall BEFORE content)', silent);
await run('content then stall (stepMs AFTER commit)', contentThenStall);
process.exit(0);
