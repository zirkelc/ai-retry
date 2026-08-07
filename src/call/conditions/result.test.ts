import { describe, expect, it } from 'vitest';
import {
  buildCallEmbeddingResultContext,
  buildCallErrorContext,
  buildCallImageResultContext,
  buildCallResultContext,
  callEmbedManyResult,
  callEmbedResult,
  callGenerateTextResult,
  callImageResult,
  callStreamTextResult,
  MockEmbeddingModel,
  MockImageModel,
  MockLanguageModel,
} from '../../internal/test-utils.js';
import { isEmbedResult, isGenerateTextResult } from '../guards.js';
import {
  createCallLanguageModelResultAPI,
  createCallResultAPI,
} from './result.js';

const { result, finishReason } =
  createCallLanguageModelResultAPI<MockLanguageModel>();
const { result: embeddingResult } = createCallResultAPI<MockEmbeddingModel>();
const { result: imageResult } = createCallResultAPI<MockImageModel>();

describe('result (call layer)', () => {
  it('should run the predicate against the current result', async () => {
    // Arrange
    const cond = result<never, MockLanguageModel>(
      (res) => isGenerateTextResult(res) && res.text === 'hi',
    );

    // Act
    const matched = await cond.evaluate(
      buildCallResultContext(await callGenerateTextResult('hi')),
    );
    const missed = await cond.evaluate(
      buildCallResultContext(await callGenerateTextResult('bye')),
    );

    // Assert
    expect(matched).toBe(true);
    expect(missed).toBe(false);
  });

  it('should return false on error attempts', async () => {
    // Arrange
    const cond = result<never, MockLanguageModel>(() => true);

    // Act
    const matched = await cond.evaluate(
      buildCallErrorContext(new Error('boom')),
    );

    // Assert
    expect(matched).toBe(false);
  });

  it('should support async predicates', async () => {
    // Arrange
    const cond = result<never, MockLanguageModel>(async () =>
      Promise.resolve(true),
    );

    // Act
    const matched = await cond.evaluate(
      buildCallResultContext(await callGenerateTextResult()),
    );

    // Assert
    expect(matched).toBe(true);
  });

  it('should pass the context as the second argument', async () => {
    // Arrange
    const seen: Array<unknown> = [];
    const cond = result<never, MockLanguageModel>((_res, ctx) => {
      seen.push(ctx.current.type, ctx.attempts.length);
      return true;
    });

    // Act
    await cond.evaluate(buildCallResultContext(await callGenerateTextResult()));

    // Assert
    expect(seen).toEqual(['result', 1]);
  });

  it('should hand over the entry point result, not a provider one', async () => {
    // Arrange — `text` is the SDK's flat field; a provider result has `content`.
    const seen: Array<unknown> = [];
    const cond = result<never, MockLanguageModel>((res) => {
      seen.push(res.operation, isGenerateTextResult(res) ? res.text : null);
      return true;
    });

    // Act
    await cond.evaluate(
      buildCallResultContext(await callGenerateTextResult('spoken')),
    );

    // Assert
    expect(seen).toEqual(['generateText', 'spoken']);
  });

  it('should judge a contentless stream result', async () => {
    // Arrange
    const cond = result<never, MockLanguageModel>(
      (res) => res.operation === 'streamText',
    );

    // Act
    const matched = await cond.evaluate(
      buildCallResultContext(callStreamTextResult('content-filter')),
    );

    // Assert
    expect(matched).toBe(true);
  });
});

describe('finishReason (call layer)', () => {
  it('should match a single reason', async () => {
    // Arrange
    const cond = finishReason<MockLanguageModel>('stop');

    // Act
    const matched = await cond.evaluate(
      buildCallResultContext(await callGenerateTextResult()),
    );

    // Assert
    expect(matched).toBe(true);
  });

  it('should match any of several reasons', async () => {
    // Arrange
    const cond = finishReason<MockLanguageModel>('content-filter', 'stop');

    // Act
    const matched = await cond.evaluate(
      buildCallResultContext(await callGenerateTextResult()),
    );

    // Assert
    expect(matched).toBe(true);
  });

  it('should not match a different reason', async () => {
    // Arrange
    const cond = finishReason<MockLanguageModel>('content-filter');

    // Act
    const matched = await cond.evaluate(
      buildCallResultContext(await callGenerateTextResult()),
    );

    // Assert
    expect(matched).toBe(false);
  });

  it('should return false on error attempts', async () => {
    // Arrange
    const cond = finishReason<MockLanguageModel>('stop');

    // Act
    const matched = await cond.evaluate(
      buildCallErrorContext(new Error('boom')),
    );

    // Assert
    expect(matched).toBe(false);
  });

  it('should read the reason off a stream result too', async () => {
    // Arrange — the field is common to both members, so no guard is needed.
    const cond = finishReason<MockLanguageModel>('content-filter');

    // Act
    const matched = await cond.evaluate(
      buildCallResultContext(callStreamTextResult('content-filter')),
    );

    // Assert
    expect(matched).toBe(true);
  });

  it('should be reachable as result.finishReason', async () => {
    // Arrange
    const cond = result.finishReason<MockLanguageModel>('stop');

    // Act
    const matched = await cond.evaluate(
      buildCallResultContext(await callGenerateTextResult()),
    );

    // Assert
    expect(matched).toBe(true);
  });
});

describe('result (other families)', () => {
  it('should judge an embed result', async () => {
    // Arrange
    const cond = embeddingResult<MockEmbeddingModel>(
      (res) => isEmbedResult(res) && res.embedding.every((n) => n === 0),
    );

    // Act
    const matched = await cond.evaluate(
      buildCallEmbeddingResultContext(await callEmbedResult([0, 0, 0])),
    );
    const missed = await cond.evaluate(
      buildCallEmbeddingResultContext(await callEmbedResult([0.1, 0.2, 0.3])),
    );

    // Assert
    expect(matched).toBe(true);
    expect(missed).toBe(false);
  });

  it('should judge an embedMany result through the same export', async () => {
    // Arrange
    const cond = embeddingResult<MockEmbeddingModel>(
      (res) => res.operation === 'embedMany',
    );

    // Act
    const matched = await cond.evaluate(
      buildCallEmbeddingResultContext(await callEmbedManyResult()),
    );

    // Assert
    expect(matched).toBe(true);
  });

  it('should return false on error attempts, as the language one does', async () => {
    // Arrange — the generic factory has its own copy of the guard.
    const cond = embeddingResult<MockEmbeddingModel>(() => true);

    // Act
    const matched = await cond.evaluate(
      buildCallErrorContext(new Error('boom')) as never,
    );

    // Assert
    expect(matched).toBe(false);
  });

  it('should judge an image result with no guard', async () => {
    // Arrange — one entry point, one union member.
    const cond = imageResult<MockImageModel>((res) => res.images.length < 2);

    // Act
    const matched = await cond.evaluate(
      buildCallImageResultContext(await callImageResult(1)),
    );
    const missed = await cond.evaluate(
      buildCallImageResultContext(await callImageResult(2)),
    );

    // Assert
    expect(matched).toBe(true);
    expect(missed).toBe(false);
  });
});
