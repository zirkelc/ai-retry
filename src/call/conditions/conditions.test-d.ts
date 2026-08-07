import { tool } from 'ai';
import { describe, expectTypeOf, it } from 'vitest';
import { z } from 'zod';
import {
  finishReason as modelFinishReason,
  httpStatus as modelHttpStatus,
} from '../../language-model/conditions/index.js';
import { createRetryableModel } from '../../language-model/create-retryable-model.js';
import { MockLanguageModel } from '../../internal/test-utils.js';
import {
  result as embeddingResult,
  isEmbedManyResult,
  isEmbedResult,
} from '../embedding-model/conditions/index.js';
import { result as imageResult } from '../image-model/conditions/index.js';
import {
  and,
  finishReason,
  or,
  httpStatus,
  isGenerateTextResult,
  isStreamTextResult,
  result,
} from '../language-model/conditions/index.js';
import type { CallRetryable } from '../types.js';
import type { ModelRetryable, ResolvableLanguageModel } from '../../types.js';
import { retryableEmbed } from '../embedding-model/functions/embed.js';
import { retryableGenerateText } from '../language-model/functions/generate-text.js';

const model = MockLanguageModel.from();

const tools = {
  weather: tool({
    description: 'get the weather',
    inputSchema: z.object({ city: z.string() }),
  }),
};

describe('the two layers are kept apart', () => {
  it('should reject a model-level condition in a call-level retry', () => {
    retryableGenerateText({
      model,
      prompt: 'hi',
      // @ts-expect-error — a model-level condition judges the provider's
      // result, which a call-level retry never produces.
      retry: [modelHttpStatus(529).switch({ model })],
    });
  });

  it('should reject a model-level result condition in a call-level retry', () => {
    retryableGenerateText({
      model,
      prompt: 'hi',
      // @ts-expect-error — same, for the result side.
      retry: [modelFinishReason('content-filter').switch({ model })],
    });
  });

  it('should reject a call-level condition in a model-level retries list', () => {
    createRetryableModel({
      model,
      // @ts-expect-error — a call-level condition judges the entry point's
      // result, which a retryable model never produces.
      retries: [httpStatus(529).switch({ model })],
    });
  });

  it('should reject a call-level result condition in a model-level retries list', () => {
    createRetryableModel({
      model,
      // @ts-expect-error — same, for the result side.
      retries: [finishReason('content-filter').switch({ model })],
    });
  });

  it('should reject a combinator that mixed the two layers', () => {
    // `and` itself accepts the mix — it infers the layer as either — but what
    // comes out belongs to neither list, so the mistake surfaces here.
    retryableGenerateText({
      model,
      prompt: 'hi',
      // @ts-expect-error — half of this condition judges the wrong layer.
      retry: [and(httpStatus(529), modelHttpStatus(529)).switch({ model })],
    });
  });

  it('should produce each layer own retryable from switch and retry', () => {
    // The terminal actions follow the layer, which is what makes the rejections
    // above possible in the first place.
    expectTypeOf(httpStatus(529).switch({ model })).toEqualTypeOf<
      CallRetryable<ResolvableLanguageModel>
    >();
    expectTypeOf(httpStatus(529).retry({ maxAttempts: 2 })).toEqualTypeOf<
      CallRetryable<ResolvableLanguageModel>
    >();
    expectTypeOf(modelHttpStatus(529).switch({ model })).toEqualTypeOf<
      ModelRetryable<ResolvableLanguageModel, never>
    >();
  });

  it('should keep a combinator on the layer its arguments came from', () => {
    expectTypeOf(
      and(httpStatus(529), finishReason('stop')).switch({ model }),
    ).toEqualTypeOf<CallRetryable<ResolvableLanguageModel>>();
    expectTypeOf(
      or(modelHttpStatus(529), modelFinishReason('stop')).switch({ model }),
    ).toEqualTypeOf<ModelRetryable<ResolvableLanguageModel, never>>();
  });

  it('should accept a call-level condition in a call-level retry', () => {
    retryableGenerateText({
      model,
      prompt: 'hi',
      retry: [and(httpStatus(529), finishReason('stop')).switch({ model })],
    });
  });
});

describe('the language-model result union', () => {
  it('should read the fields both entry points share without narrowing', () => {
    result((res) => {
      expectTypeOf(res.operation).toEqualTypeOf<
        'generateText' | 'streamText'
      >();
      expectTypeOf(res.finishReason).not.toBeAny();
      expectTypeOf(res.usage).not.toBeAny();
      return true;
    });
  });

  it('should reject a field only one entry point has', () => {
    result((res) => {
      // @ts-expect-error — `text` exists on a `generateText` result only, so it
      // needs a guard.
      return res.text === '';
    });
  });

  it('should narrow to the generateText member through a guard', () => {
    result((res) => {
      if (!isGenerateTextResult(res)) return false;
      expectTypeOf(res.text).toEqualTypeOf<string>();
      return true;
    });
  });

  it('should narrow to the streamText member through a guard', () => {
    result((res) => {
      if (!isStreamTextResult(res)) return false;
      expectTypeOf(res.operation).toEqualTypeOf<'streamText'>();
      return true;
    });
  });

  it('should narrow on the discriminant directly', () => {
    result((res) => {
      if (res.operation !== 'generateText') return false;
      expectTypeOf(res.text).toEqualTypeOf<string>();
      return true;
    });
  });

  it('should type the tool calls against the tool set named at the condition', () => {
    // A tool call is static or dynamic, and only a static one has a known
    // name — the same discrimination a direct `generateText` call requires.
    result<typeof tools>((res) => {
      if (!isGenerateTextResult(res)) return false;
      const call = res.toolCalls[0]!;
      if (call.dynamic) return false;
      expectTypeOf(call.toolName).toEqualTypeOf<'weather'>();
      expectTypeOf(call.input).toEqualTypeOf<{ city: string }>();
      return true;
    });
  });

  it('should leave the tool calls at the bound when no tool set is named', () => {
    result((res) => {
      if (!isGenerateTextResult(res)) return false;
      const call = res.toolCalls[0]!;
      if (call.dynamic) return false;
      expectTypeOf(call.toolName).toEqualTypeOf<string>();
      return true;
    });
  });
});

describe('the embedding-model result union', () => {
  it('should reject a field only one entry point has', () => {
    embeddingResult((res) => {
      // @ts-expect-error — `embedding` is `embed`'s; `embedMany` has
      // `embeddings`.
      return res.embedding.length === 0;
    });
  });

  it('should narrow embed and embedMany apart from one export', () => {
    embeddingResult((res) => {
      if (isEmbedResult(res)) {
        expectTypeOf(res.embedding).toEqualTypeOf<Array<number>>();
        return true;
      }
      if (isEmbedManyResult(res)) {
        expectTypeOf(res.embeddings).toEqualTypeOf<Array<Array<number>>>();
        return true;
      }
      return false;
    });
  });

  it('should reject a language-model guard on an embedding result', () => {
    embeddingResult((res) => {
      // @ts-expect-error — wrong family.
      return isGenerateTextResult(res);
    });
  });
});

describe('the image-model result', () => {
  it('should read its fields with no guard, having a single member', () => {
    imageResult((res) => {
      expectTypeOf(res.operation).toEqualTypeOf<'generateImage'>();
      return res.images.length === 0;
    });
  });
});

describe('result conditions plug into their own entry points', () => {
  it('should accept an embedding result condition in retryableEmbed', () => {
    retryableEmbed({
      model: MockLanguageModel.from() as never,
      value: 'hi',
      retry: [
        embeddingResult(() => true).switch({
          model: 'openai/text-embedding-3-small',
        }),
      ],
    });
  });

  it('should reject a language result condition in retryableEmbed', () => {
    retryableEmbed({
      model: MockLanguageModel.from() as never,
      value: 'hi',
      // @ts-expect-error — wrong family.
      retry: [result(() => true).switch({ model })],
    });
  });
});
