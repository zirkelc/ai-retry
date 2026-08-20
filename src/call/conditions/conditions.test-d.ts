import { tool } from 'ai';
import { describe, expectTypeOf, it } from 'vitest';
import { z } from 'zod';
import {
  MockEmbeddingModel,
  MockLanguageModel,
} from '../../internal/test-utils.js';
import {
  finishReason as modelFinishReason,
  httpStatus as modelHttpStatus,
} from '../../model/language-model/conditions/index.js';
import { createRetryableModel } from '../../model/language-model/create-retryable-model.js';
import type {
  LanguageModel,
  ModelRetryable,
  ResolvableLanguageModel,
} from '../../types.js';
import { retryableEmbed } from '../embed/embed.js';
import { result as embedResult } from '../embed/conditions/index.js';
import { retryableEmbedMany } from '../embed-many/embed-many.js';
import { result as embedManyResult } from '../embed-many/conditions/index.js';
import { result as imageResult } from '../generate-image/conditions/index.js';
import { retryableGenerateText } from '../generate-text/generate-text.js';
import {
  and,
  finishReason,
  httpStatus,
  or,
  result,
} from '../generate-text/conditions/index.js';
import { retryableStreamText } from '../stream-text/stream-text.js';
import { result as streamResult } from '../stream-text/conditions/index.js';
import type { GenerateTextCommitResult } from '../generate-text/types.js';
import type { StreamTextCommitResult } from '../stream-text/types.js';
import type {
  CallRetryable,
  CallRetryAttempt,
  CallRetryContext,
} from '../types.js';

const model = MockLanguageModel.from();
const embeddingModel = MockEmbeddingModel.from();

/**
 * Built up front so each `@ts-expect-error` below sits directly above the line
 * it is about: a retryable spelled inline wraps, and the directive would then
 * cover the wrong line.
 */
const readsGeneratedText = result((res) => res.text === '').switch({ model });
const readsOneEmbedding = embedResult(
  (res) => res.embedding.length === 0,
).switch({ model: embeddingModel });
const readsManyEmbeddings = embedManyResult(
  (res) => res.embeddings.length === 0,
).switch({ model: embeddingModel });

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

describe('the entry points are kept apart', () => {
  it('should reject a generateText result condition in retryableStreamText', () => {
    // This is the direction that would break at runtime: a pre-commit stream
    // has no `text` and no `toolCalls` to read, by construction.
    retryableStreamText({
      model,
      prompt: 'hi',
      // @ts-expect-error — reads fields a stream cannot have produced yet.
      retry: [readsGeneratedText],
    });
  });

  it('should reject an embed result condition in retryableEmbedMany', () => {
    retryableEmbedMany({
      model: embeddingModel,
      values: ['hi'],
      // @ts-expect-error — `embedding` is `embed`'s; `embedMany` produces
      // `embeddings`.
      retry: [readsOneEmbedding],
    });
  });

  it('should reject an embedMany result condition in retryableEmbed', () => {
    retryableEmbed({
      model: embeddingModel,
      value: 'hi',
      // @ts-expect-error — the mirror image of the above.
      retry: [readsManyEmbeddings],
    });
  });

  it('should reject a language result condition in retryableEmbed', () => {
    retryableEmbed({
      model: embeddingModel,
      value: 'hi',
      // @ts-expect-error — wrong family as well as wrong entry point.
      retry: [readsGeneratedText],
    });
  });

  it('should accept each entry point own result condition', () => {
    retryableGenerateText({
      model,
      prompt: 'hi',
      retry: [result((res) => res.text === '').switch({ model })],
    });
    retryableStreamText({
      model,
      prompt: 'hi',
      retry: [
        streamResult((res) => res.finishReason === 'length').switch({ model }),
      ],
    });
    retryableEmbed({
      model: embeddingModel,
      value: 'hi',
      retry: [readsOneEmbedding],
    });
    retryableEmbedMany({
      model: embeddingModel,
      values: ['hi'],
      retry: [readsManyEmbeddings],
    });
  });

  it('should accept an error condition anywhere, reading no result at all', () => {
    // The permissive end of `COMMIT`: nothing is read off the result, so there
    // is no entry point the condition cannot be written into.
    retryableGenerateText({
      model,
      prompt: 'hi',
      retry: [httpStatus(529).switch({ model })],
    });
    retryableStreamText({
      model,
      prompt: 'hi',
      retry: [httpStatus(529).switch({ model })],
    });
  });
});

describe('each entry point result reads without narrowing', () => {
  it('should type a generateText result as the completed generation', () => {
    result((res) => {
      expectTypeOf(res.text).toEqualTypeOf<string>();
      expectTypeOf(res.finishReason).not.toBeAny();
      expectTypeOf(res.usage).not.toBeAny();
      return true;
    });
  });

  it('should type a streamText result as what a contentless stream reports', () => {
    streamResult((res) => {
      expectTypeOf(res.finishReason).not.toBeAny();
      expectTypeOf(res.usage).not.toBeAny();
      // @ts-expect-error — no content exists before the commit point.
      return res.text === '';
    });
  });

  it('should type an embed result as a single embedding', () => {
    embedResult((res) => {
      expectTypeOf(res.embedding).toEqualTypeOf<Array<number>>();
      // @ts-expect-error — that is `embedMany`'s.
      return res.embeddings.length === 0;
    });
  });

  it('should type an embedMany result as many embeddings', () => {
    embedManyResult((res) => {
      expectTypeOf(res.embeddings).toEqualTypeOf<Array<Array<number>>>();
      // @ts-expect-error — that is `embed`'s.
      return res.embedding.length === 0;
    });
  });

  it('should type an image result with its images readable directly', () => {
    imageResult((res) => {
      expectTypeOf(res.images).not.toBeAny();
      return res.images.length === 0;
    });
  });

  it('should evaluate against a context carrying the same commit result', () => {
    // `LayerContext` is reachable from outside only here, through `evaluate`:
    // each conditions module writes its predicate's `ctx` type out by hand, so
    // nothing else pins the mapping. Without this, `COMMIT` could be dropped
    // from `LayerContext` and the whole suite would still pass.
    const cond = result((res) => res.text === '');

    expectTypeOf<Parameters<typeof cond.evaluate>[0]>().toEqualTypeOf<
      CallRetryContext<ResolvableLanguageModel, GenerateTextCommitResult>
    >();
  });

  it('should carry the commit result into the context, not only the result', () => {
    // The predicate's second argument reaches the same attempts the first was
    // taken from. Each conditions module declares this by hand, so it can drift
    // from the result parameter beside it; every other assertion here reads
    // only the result and would not notice.
    result((_res, ctx) => {
      const current = ctx.current;
      if (current.type !== 'result') return false;
      expectTypeOf(current.result).toEqualTypeOf<GenerateTextCommitResult>();
      expectTypeOf(ctx.attempts).toEqualTypeOf<
        Array<CallRetryAttempt<LanguageModel, GenerateTextCommitResult>>
      >();
      return true;
    });

    streamResult((_res, ctx) => {
      const current = ctx.current;
      if (current.type !== 'result') return false;
      expectTypeOf(current.result).toEqualTypeOf<StreamTextCommitResult>();
      return true;
    });
  });

  it('should type the tool calls against the tool set named at the condition', () => {
    // A tool call is static or dynamic, and only a static one has a known
    // name — the same discrimination a direct `generateText` call requires.
    result<typeof tools>((res) => {
      const call = res.toolCalls[0]!;
      if (call.dynamic) return false;
      expectTypeOf(call.toolName).toEqualTypeOf<'weather'>();
      expectTypeOf(call.input).toEqualTypeOf<{ city: string }>();
      return true;
    });
  });

  it('should leave the tool calls at the bound when no tool set is named', () => {
    result((res) => {
      const call = res.toolCalls[0]!;
      if (call.dynamic) return false;
      expectTypeOf(call.toolName).toEqualTypeOf<string>();
      return true;
    });
  });
});
