/**
 * Deterministic (no API key) demonstration of recovering a *canned refusal*
 * from a streaming call with `createRetryableStream` + `refusalGate`.
 *
 * Some content-filter false positives don't surface as an error or a
 * `content-filter` finish reason — the model just streams a natural-language
 * refusal ("I'm sorry, but I cannot assist...") and finishes with `stop`. To a
 * stream retry that commits on the first content part, that refusal IS the
 * committed answer, so it can never fail over from it.
 *
 * `refusalGate` moves the text-commit boundary later: it buffers the leading
 * `text-delta` parts (nothing has reached the caller yet) and only commits once
 * the text diverges from every known refusal phrase. If the buffer instead
 * *matches* a phrase, it throws a `RefusalError` and the call fails over to
 * another model. A real answer that merely shares a leading fragment ("I'm
 * sorry to hear that...") diverges within a delta or two and commits normally.
 *
 * Run:
 *   pnpm build && pnpm tsx examples/retryable-stream-refusal-sim.ts
 */
import { createOpenAI } from '@ai-sdk/openai';
import { streamText } from 'ai';
import {
  createRetryableStream,
  RefusalError,
  refusalGate,
  type RetryableStreamOptions,
} from 'ai-retry/experimental/stream';
import { error } from 'ai-retry/language-model/conditions';

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

/** Stream the given content deltas, then finish with `stop`. */
const bodyOf = (deltas: Array<string>) =>
  new ReadableStream<Uint8Array>({
    start(controller) {
      controller.enqueue(chunk({ role: 'assistant' }));
      for (const delta of deltas) controller.enqueue(chunk({ content: delta }));
      controller.enqueue(chunk({}, 'stop'));
      controller.enqueue(encoder.encode('data: [DONE]\n\n'));
      controller.close();
    },
  });

/** Known false-positive refusals worth failing over from. */
const REFUSALS = [
  "I'm sorry, but I cannot assist",
  'I cannot help with that request',
];

const retryableStreamText = (
  options: RetryableStreamOptions,
  args: Omit<Parameters<typeof streamText>[0], 'model'>,
) => {
  const run = createRetryableStream(options);
  return run(
    (attempt) =>
      streamText({
        ...args,
        model: attempt.model,
        abortSignal: attempt.abortSignal,
        onError: () => {},
      } as Parameters<typeof streamText>[0]),
    { abortSignal: args.abortSignal },
  );
};

/**
 * Attempt 1 streams `firstDeltas`; the fallback always answers cleanly. Prints
 * how many attempts ran and the text the caller ends up with.
 */
const scenario = async (label: string, firstDeltas: Array<string>) => {
  let attempt = 0;
  const openai = createOpenAI({
    apiKey: 'sim',
    fetch: async () => {
      attempt += 1;
      return sseResponse(
        bodyOf(
          attempt === 1
            ? firstDeltas
            : ['Chlorine tablets ', 'sanitize pool water.'],
        ),
      );
    },
  });

  const result = await retryableStreamText(
    {
      model: openai.chat('gpt-4o-mini'),
      retries: [
        error((e) => e instanceof RefusalError).switch({
          model: openai.chat('gpt-4o'),
        }),
      ],
      commitGate: refusalGate(REFUSALS),
    },
    { prompt: 'What sanitizes pool water?' },
  );

  let text = '';
  for await (const delta of result.textStream) text += delta;
  console.log(`\n=== ${label} ===`);
  console.log(`   attempts: ${attempt}`);
  console.log(`   text: "${text}"`);
};

/** A canned refusal split across deltas — recovered by failing over. */
await scenario('canned refusal (fail over)', [
  "I'm sorry, ",
  'but I cannot assist',
  ' with that request.',
]);

/** A real answer that shares a leading fragment — committed, no false retry. */
await scenario('real answer sharing a fragment (no retry)', [
  "I'm sorry ",
  'to hear that! Chlorine tablets work well.',
]);

console.log(
  '\nTakeaway: refusalGate buffers only the leading text, so a canned refusal\nfails over while a genuine answer commits after diverging from the phrase.',
);
process.exit(0);
