/**
 * Repro for the AI SDK feature request: streaming `timeout` has no deadline
 * covering the "headers received but no content yet" window.
 *
 * A custom `fetch` returns 200 immediately, emits the role prelude plus an SSE
 * keep-alive comment, then goes silent — never producing a content chunk.
 *
 *   chunkMs -> never fires (only measures gaps BETWEEN content chunks) -> HANG
 *   totalMs -> fires, but only at its coarse bound (parts=[start, abort])
 *
 * Run (ai@7, @ai-sdk/openai@4):
 *   pnpm tsx examples/repro-firstchunk-timeout.ts
 */
import { createOpenAI } from '@ai-sdk/openai';
import { streamText } from 'ai';

const enc = new TextEncoder();

/** 200 OK, role prelude + keep-alive comment, then silence. No content chunk. */
const preludeThenStall = (signal?: AbortSignal) =>
  new ReadableStream<Uint8Array>({
    start(c) {
      c.enqueue(
        enc.encode(
          `data: ${JSON.stringify({ id: 'x', object: 'chat.completion.chunk', choices: [{ index: 0, delta: { role: 'assistant' }, finish_reason: null }] })}\n\n`,
        ),
      );
      c.enqueue(enc.encode(': keep-alive ping\n\n')); // SSE comment: resets any byte-level timer
      signal?.addEventListener('abort', () => {
        try {
          c.error(signal.reason);
        } catch {}
      });
    },
  });

const openai = createOpenAI({
  apiKey: 'test',
  fetch: async (_i, init) =>
    new Response(preludeThenStall(init?.signal ?? undefined), {
      status: 200,
      headers: { 'content-type': 'text/event-stream' },
    }),
});

const run = async (label: string, opts: Record<string, unknown>) => {
  const result = streamText({
    model: openai.chat('gpt-4o'),
    prompt: 'hi',
    onError: () => {},
    ...opts,
  } as Parameters<typeof streamText>[0]);

  const t0 = Date.now();
  const consume = (async () => {
    const parts: string[] = [];
    try {
      for await (const p of result.fullStream) parts.push(p.type);
    } catch (e) {
      return `aborted@${Date.now() - t0}ms (${(e as Error).name})`;
    }
    return `ended@${Date.now() - t0}ms parts=[${parts}]`;
  })();
  const guard = new Promise<string>((r) =>
    setTimeout(
      () => r('HANG (nothing in 2000ms) — deadline never fired'),
      2000,
    ),
  );
  console.log(`${label.padEnd(12)} -> ${await Promise.race([consume, guard])}`);
};

await run('chunkMs:500', { timeout: { chunkMs: 500 } });
await run('totalMs:500', { timeout: { totalMs: 500 } });
process.exit(0);
