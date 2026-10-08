import { type LanguageModelUsage, NoObjectGeneratedError } from 'ai';
import { describe, expect, it } from 'vitest';
import {
  buildCallErrorContext,
  buildCallResultContext,
  callEmbedResult,
  callGenerateTextResult,
  MockLanguageModel,
} from '../../internal/test-utils.js';
import type { CallFinishReason } from '../types.js';
import {
  createFinishReasonAPI,
  errorFinishReason,
  resultFinishReason,
} from './finish-reason.js';

const { finishReason } = createFinishReasonAPI<MockLanguageModel>();

/** The error the SDK raises when a structured output cannot be parsed. */
const noObjectError = (finishReason: CallFinishReason | undefined) =>
  new NoObjectGeneratedError({
    response: { id: 'id-0', timestamp: new Date(0), modelId: 'mock-model-id' },
    usage: {} as LanguageModelUsage,
    finishReason: finishReason as CallFinishReason,
  });

describe('resultFinishReason', () => {
  it('should read the finish reason of a language result', async () => {
    // Arrange
    const result = await callGenerateTextResult();

    // Act
    const reason = resultFinishReason(result);

    // Assert
    expect(reason).toBe('stop');
  });

  it('should report nothing for a result without a finish reason', async () => {
    // Arrange: embeddings have no notion of one.
    const result = await callEmbedResult();

    // Act
    const reason = resultFinishReason(result);

    // Assert
    expect(reason).toBeUndefined();
  });
});

describe('errorFinishReason', () => {
  it('should read the finish reason an SDK error carries', () => {
    // Arrange
    const error = noObjectError('content-filter');

    // Act
    const reason = errorFinishReason(error);

    // Assert
    expect(reason).toBe('content-filter');
  });

  it('should read a normal finish reason as it was reported', () => {
    // Arrange: reading does not judge; the condition decides what counts.
    const error = noObjectError('stop');

    // Act
    const reason = errorFinishReason(error);

    // Assert
    expect(reason).toBe('stop');
  });

  it('should report nothing for an SDK error without a finish reason', () => {
    // Arrange
    const error = noObjectError(undefined);

    // Act
    const reason = errorFinishReason(error);

    // Assert
    expect(reason).toBeUndefined();
  });

  it('should report nothing for a foreign error with a finish reason field', () => {
    // Arrange
    const error = Object.assign(new Error('boom'), {
      finishReason: 'content-filter',
    });

    // Act
    const reason = errorFinishReason(error);

    // Assert
    expect(reason).toBeUndefined();
  });

  it.each([undefined, null, 'content-filter', { finishReason: 'length' }])(
    'should report nothing for the non-error value %j',
    (value) => {
      // Act
      const reason = errorFinishReason(value);

      // Assert
      expect(reason).toBeUndefined();
    },
  );
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

  it('should return false on an error that carries no finish reason', async () => {
    // Arrange
    const cond = finishReason<MockLanguageModel>('stop');

    // Act
    const matched = await cond.evaluate(
      buildCallErrorContext(new Error('boom')),
    );

    // Assert
    expect(matched).toBe(false);
  });

  it('should match an SDK error by the finish reason it carries', async () => {
    // Arrange
    const cond = finishReason<MockLanguageModel>('content-filter');

    // Act
    const matched = await cond.evaluate(
      buildCallErrorContext(noObjectError('content-filter')),
    );

    // Assert
    expect(matched).toBe(true);
  });

  it('should not match an SDK error carrying another finish reason', async () => {
    // Arrange
    const cond = finishReason<MockLanguageModel>('content-filter');

    // Act
    const matched = await cond.evaluate(
      buildCallErrorContext(noObjectError('length')),
    );

    // Assert
    expect(matched).toBe(false);
  });

  it.each(['stop', 'tool-calls'] as const)(
    "should not match an SDK error carrying the normal finish reason '%s'",
    async (reason) => {
      // Arrange: the generation finished normally and its content was
      // rejected, so the finish reason does not explain the failure.
      const cond = finishReason<MockLanguageModel>(reason);

      // Act
      const matched = await cond.evaluate(
        buildCallErrorContext(noObjectError(reason)),
      );

      // Assert
      expect(matched).toBe(false);
    },
  );

  it('should not match an SDK error whose finish reason is missing', async () => {
    // Arrange: the SDK declares the field optional on this error.
    const cond = finishReason<MockLanguageModel>('content-filter');

    // Act
    const matched = await cond.evaluate(
      buildCallErrorContext(noObjectError(undefined)),
    );

    // Assert
    expect(matched).toBe(false);
  });

  it('should not match a foreign error that has a finish reason field', async () => {
    // Arrange: only SDK errors are read, whatever fields another error has.
    const cond = finishReason<MockLanguageModel>('content-filter');
    const foreign = Object.assign(new Error('boom'), {
      finishReason: 'content-filter',
    });

    // Act
    const matched = await cond.evaluate(buildCallErrorContext(foreign));

    // Assert
    expect(matched).toBe(false);
  });
});
