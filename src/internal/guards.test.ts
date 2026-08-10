import type { TextStreamPart, ToolSet } from 'ai';
import { streamText } from 'ai';
import { describe, expect, it } from 'vitest';
import type { LanguageModelStreamPart } from '../types.js';
import { isStreamContentPart } from './guards.js';
import { MockLanguageModel, Streams } from './test-utils.js';

/**
 * `isStreamContentPart` has to agree with the AI SDK's own notion of generated
 * output, because the commit boundary is only defensible if it is the same
 * boundary the SDK draws. Disagreeing in one direction commits an attempt that
 * produced nothing, in the other keeps failing over after the caller has
 * already seen content.
 *
 * The SDK's classifier (`isOutputChunk`) is internal and not exported, so it
 * cannot be imported and compared. It is *observable*, though: the SDK arms a
 * `firstChunkMs` deadline when the response stream starts and clears it on the
 * first output chunk. So a part that stops the deadline firing is output to the
 * SDK, and one that does not is framing.
 *
 * Reading the answer out of the SDK rather than restating its table here is the
 * point of this file. A copied table agrees with whatever the SDK did on the
 * day it was copied; this fails when the SDK reclassifies a part or adds one.
 */

/** Long enough that the deadline below fires well before the stream ends. */
const STALL_MS = 300;
const FIRST_CHUNK_MS = 50;

/**
 * Emit `parts` immediately, then stall past the first-chunk deadline.
 *
 * Resolves true when the deadline never fired, meaning something in `parts`
 * counted as output to the SDK.
 */
const clearsFirstChunkDeadline = async (
  parts: Array<LanguageModelStreamPart>,
): Promise<boolean> => {
  const stream = new ReadableStream<LanguageModelStreamPart>({
    async start(controller) {
      controller.enqueue({ type: 'stream-start', warnings: [] });
      for (const part of parts) controller.enqueue(part);
      await new Promise((resolve) => setTimeout(resolve, STALL_MS));
      controller.enqueue({
        type: 'finish',
        finishReason: { unified: 'stop', raw: 'stop' },
        usage: {
          inputTokens: { total: 1, noCache: 1, cacheRead: 0, cacheWrite: 0 },
          outputTokens: { total: 1, text: 1, reasoning: 0 },
        },
      });
      controller.close();
    },
  });

  const model = MockLanguageModel.from({ doStream: async () => ({ stream }) });
  const result = streamText({
    model,
    prompt: 'Hello!',
    timeout: { firstChunkMs: FIRST_CHUNK_MS },
    /** The deadline surfaces as a part; nothing should be logged for it. */
    onError: () => {},
  });

  const emitted = await Streams.toArray(result.fullStream);
  return !emitted.some(
    (part) => part.type === 'abort' || part.type === 'error',
  );
};

/**
 * Every part a provider stream can carry that is worth classifying.
 *
 * `prefix` opens the block the part belongs to, since the SDK drops an orphan
 * delta. It is probed on its own too, so that a cleared deadline is attributed
 * to `part` rather than to whatever had to precede it.
 */
const PARTS: Array<{
  name: string;
  prefix: Array<LanguageModelStreamPart>;
  part: LanguageModelStreamPart;
}> = [
  {
    name: 'text-delta',
    prefix: [{ type: 'text-start', id: '1' }],
    part: { type: 'text-delta', id: '1', delta: 'hi' },
  },
  {
    name: 'reasoning-delta',
    prefix: [{ type: 'reasoning-start', id: '1' }],
    part: { type: 'reasoning-delta', id: '1', delta: 'hm' },
  },
  {
    name: 'tool-input-delta',
    prefix: [{ type: 'tool-input-start', id: '1', toolName: 'lookup' }],
    part: { type: 'tool-input-delta', id: '1', delta: '{}' },
  },
  {
    name: 'tool-input-start',
    prefix: [],
    part: { type: 'tool-input-start', id: '1', toolName: 'lookup' },
  },
  { name: 'text-start', prefix: [], part: { type: 'text-start', id: '1' } },
  {
    name: 'reasoning-start',
    prefix: [],
    part: { type: 'reasoning-start', id: '1' },
  },
  { name: 'raw', prefix: [], part: { type: 'raw', rawValue: { a: 1 } } },
  {
    name: 'response-metadata',
    prefix: [],
    part: {
      type: 'response-metadata',
      id: 'id-0',
      modelId: 'mock-model-id',
      timestamp: new Date(0),
    },
  },
];

describe('isStreamContentPart', () => {
  it.each(PARTS)(
    'should classify $name exactly as the SDK does',
    async ({ prefix, part }) => {
      // Arrange — attribute the outcome to `part`, not to its prefix.
      const prefixIsOutput =
        prefix.length > 0 ? await clearsFirstChunkDeadline(prefix) : false;

      // Act
      const sequenceIsOutput = await clearsFirstChunkDeadline([
        ...prefix,
        part,
      ]);
      const sdkTreatsAsOutput = sequenceIsOutput && !prefixIsOutput;

      // Assert
      expect(isStreamContentPart(part)).toBe(sdkTreatsAsOutput);
    },
  );

  it('should not commit on an empty delta, as the SDK does not', () => {
    // Arrange — a zero-length delta is a heartbeat, not content.
    const empty: LanguageModelStreamPart = {
      type: 'text-delta',
      id: '1',
      delta: '',
    };

    // Assert
    expect(isStreamContentPart(empty)).toBe(false);
  });

  it("should read the SDK's spelling of a delta as well as the provider's", () => {
    // Arrange — the same part, as the two vocabularies spell it.
    const providerPart: LanguageModelStreamPart = {
      type: 'text-delta',
      id: '1',
      delta: 'hi',
    };
    const sdkPart: TextStreamPart<ToolSet> = {
      type: 'text-delta',
      id: '1',
      text: 'hi',
    };

    // Assert
    expect(isStreamContentPart(providerPart)).toBe(true);
    expect(isStreamContentPart(sdkPart)).toBe(true);
  });

  it('should commit on the parts only the SDK stream carries', () => {
    // Arrange — a provider stream expresses these differently or not at all,
    // so they are unreachable below a model and only decide around a call.
    const toolCall = { type: 'tool-call' } as TextStreamPart<ToolSet>;
    const file = { type: 'file' } as TextStreamPart<ToolSet>;
    const reasoningFile = { type: 'reasoning-file' } as TextStreamPart<ToolSet>;

    // Assert
    expect(isStreamContentPart(toolCall)).toBe(true);
    expect(isStreamContentPart(file)).toBe(true);
    expect(isStreamContentPart(reasoningFile)).toBe(true);
  });
});
