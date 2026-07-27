/**
 * Deterministic, offline twin of `retryable-stream-fetch-intercept.ts` — same
 * `createRetryableStream` + custom-`fetch` wiring, but the "network" is a
 * synthetic OpenAI chat SSE stream, so it needs NO API key and never flakes.
 * Good for a quick local check or a CI regression test of the fail-over rule.
 *
 * It runs the two cases that define when a stream can recover:
 *
 *   A. The connection drops BEFORE any text streams -> fail-over fires, the
 *      second attempt streams the answer, the caller never sees the error.
 *   B. The connection drops AFTER some text streams  -> the attempt has already
 *      committed (a content part was emitted), so retrying would duplicate
 *      output; the error is surfaced to the caller instead. NO retry.
 *
 * The realism that makes B behave like production is timing: real provider
 * deltas arrive spaced out, so the SDK emits them as `text-delta` parts before
 * a later transport error. If every byte arrived in one synchronous burst the
 * error would pre-empt the deltas and even B would fail over — so this sim puts
 * a small gap before the simulated reset, exactly as a real socket would.
 *
 * Run:
 *   pnpm build && pnpm tsx examples/retryable-stream-offline-sim.ts
 */
import { createOpenAI } from '@ai-sdk/openai';
import { streamText } from 'ai';
import {
  createRetryableStream,
  type RetryableStreamOptions,
} from 'ai-retry/experimental/stream';

const encoder = new TextEncoder();
const sse = (data: unknown) => `data: ${JSON.stringify(data)}\n\n`;
const chunk = (delta: Record<string, unknown>, finish: string | null = null) =>
  sse({
    id: 'sim',
    object: 'chat.completion.chunk',
    model: 'gpt-4o',
    choices: [{ index: 0, delta, finish_reason: finish }],
  });

/** A complete chat-completion SSE response: role, two content deltas, stop. */
const fullResponseEvents = [
  chunk({ role: 'assistant' }),
  chunk({ content: 'Exponential backoff ' }),
  chunk({ content: 'spaces out retries.' }),
  chunk({}, 'stop'),
  'data: [DONE]\n\n',
];

type CutMode = 'before-content' | 'after-content' | 'none';

const eventHasContent = (event: string): boolean => {
  const data = event.match(/^data: (.*)$/m)?.[1];
  if (!data || data === '[DONE]') return false;
  try {
    return Boolean(JSON.parse(data)?.choices?.[0]?.delta?.content);
  } catch {
    return false;
  }
};

/**
 * Emit the response events one at a time with a small gap (a stand-in for
 * network pacing), then sever the stream per `mode` to mimic a transport reset.
 */
const simulatedBody = (mode: CutMode): ReadableStream<Uint8Array> => {
  let contentSeen = 0;
  let index = 0;
  return new ReadableStream<Uint8Array>({
    async pull(controller) {
      await new Promise((resolve) => setTimeout(resolve, 5));
      const event = fullResponseEvents[index++];
      if (event === undefined) {
        controller.close();
        return;
      }
      const isContent = eventHasContent(event);

      if (mode === 'before-content' && isContent) {
        controller.error(new Error('connection reset (pre-content)'));
        return;
      }
      controller.enqueue(encoder.encode(event));
      if (mode === 'after-content' && isContent && ++contentSeen >= 2) {
        controller.error(new Error('connection reset (post-content)'));
      }
    },
  });
};

/** A `fetch` that severs only its first call; later calls stream cleanly. */
const makeFetch = (mode: CutMode): typeof fetch => {
  let calls = 0;
  return async () => {
    const attempt = ++calls;
    const applied: CutMode = attempt === 1 ? mode : 'none';
    console.log(`   fetch #${attempt}: ${applied}`);
    return new Response(simulatedBody(applied), {
      status: 200,
      headers: { 'content-type': 'text/event-stream' },
    });
  };
};

/**
 * Drop-in `streamText` glue on top of `createRetryableStream`: re-run the whole
 * `streamText` call per attempt with the attempt's model and fresh deadline
 * signal, deciding commit/fail-over from `stream`.
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
        onError: () => {},
      } as Parameters<typeof streamText>[0]);
    },
    { abortSignal: args.abortSignal },
  );
};

const runScenario = async (label: string, mode: CutMode) => {
  console.log(`\n=== ${label} ===`);
  const openai = createOpenAI({ apiKey: 'sim', fetch: makeFetch(mode) });

  const result = await retryableStreamText(
    {
      model: openai.chat('gpt-4o-mini'),
      retries: [openai.chat('gpt-4o')],
    },
    { prompt: 'What is exponential backoff?' },
  );

  let text = '';
  try {
    for await (const delta of result.textStream) text += delta;
    console.log(`   -> RECOVERED: "${text}"`);
  } catch (error) {
    console.log(
      `   -> NOT RECOVERED after "${text}": ${(error as Error).message}`,
    );
  }
};

await runScenario(
  'A. drop BEFORE content (expect: fail-over, full answer)',
  'before-content',
);
await runScenario(
  'B. drop AFTER content (expect: no retry, error surfaces)',
  'after-content',
);

console.log(
  '\nTakeaway: a stream can only fail over before its first content part.',
);
