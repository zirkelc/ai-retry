import { describe, expect, it } from 'vitest';
import {
  buildCallEmbeddingResultContext,
  buildCallErrorContext,
  buildCallImageResultContext,
  buildCallResultContext,
  callEmbedResult,
  callGenerateTextResult,
  callImageResult,
  callStreamTextResult,
  MockEmbeddingModel,
  MockImageModel,
  MockLanguageModel,
} from '../../internal/test-utils.js';
import type { EmbedCommitResult } from '../embed/types.js';
import type { GenerateImageCommitResult } from '../generate-image/types.js';
import type { GenerateTextCommitResult } from '../generate-text/types.js';
import type { StreamTextCommitResult } from '../stream-text/types.js';
import { createCallResultAPI, createFinishReasonAPI } from './result.js';

/**
 * The shared result-side factories, exercised once here. What each entry point
 * does with them — which commit result it binds, and whether it takes
 * `finishReason` at all — is asserted beside that entry point instead.
 */

const { result } = createCallResultAPI<
  MockLanguageModel,
  GenerateTextCommitResult
>();
const { finishReason } = createFinishReasonAPI<
  MockLanguageModel,
  GenerateTextCommitResult
>();
const { result: streamResult } = createCallResultAPI<
  MockLanguageModel,
  StreamTextCommitResult
>();
const { result: embeddingResult } = createCallResultAPI<
  MockEmbeddingModel,
  EmbedCommitResult
>();
const { result: imageResult } = createCallResultAPI<
  MockImageModel,
  GenerateImageCommitResult
>();

describe('result (call layer)', () => {
  it('should run the predicate against the current result', async () => {
    // Arrange
    const cond = result<MockLanguageModel>((res) => res.text === 'hi');

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
    const cond = result<MockLanguageModel>(() => true);

    // Act
    const matched = await cond.evaluate(
      buildCallErrorContext(new Error('boom')),
    );

    // Assert
    expect(matched).toBe(false);
  });

  it('should support async predicates', async () => {
    // Arrange
    const cond = result<MockLanguageModel>(async () => Promise.resolve(true));

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
    const cond = result<MockLanguageModel>((_res, ctx) => {
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
    const cond = result<MockLanguageModel>((res) => {
      seen.push(res.text);
      return true;
    });

    // Act
    await cond.evaluate(
      buildCallResultContext(await callGenerateTextResult('spoken')),
    );

    // Assert
    expect(seen).toEqual(['spoken']);
  });

  it('should hand the result over untouched, not a copy of it', async () => {
    // Arrange — the SDK exposes most of a result through prototype getters, so
    // anything that rebuilt it on the way through would arrive with `text`
    // undefined.
    const produced = await callGenerateTextResult('spoken');
    let seen: unknown;
    const cond = result<MockLanguageModel>((res) => {
      seen = res;
      return true;
    });

    // Act
    await cond.evaluate(buildCallResultContext(produced));

    // Assert
    expect(seen).toBe(produced);
  });

  it('should judge a contentless stream result', async () => {
    // Arrange
    const cond = streamResult<MockLanguageModel>(
      (res) => res.usage.outputTokens === 0,
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
});

describe('result (other families)', () => {
  it('should judge an embed result', async () => {
    // Arrange
    const cond = embeddingResult<MockEmbeddingModel>((res) =>
      res.embedding.every((n) => n === 0),
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

  it('should judge an image result', async () => {
    // Arrange
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
