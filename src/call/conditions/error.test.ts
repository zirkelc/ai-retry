import { APICallError, NoImageGeneratedError } from 'ai';
import { describe, expect, it } from 'vitest';
import { createErrorAPI } from '../../internal/conditions/error.js';
import { createNoImageAPI } from '../../internal/conditions/no-image.js';
import {
  abortError,
  apiError,
  buildCallErrorContext,
  buildCallImageErrorContext,
  buildCallResultContext,
  callGenerateTextResult,
  Errors,
  MockImageModel,
  MockLanguageModel,
  timeoutError,
} from '../../internal/test-utils.js';

/**
 * The error API is one implementation instantiated per layer. These exercise
 * the `'call'` instantiation — the same matchers, reading a call context.
 */
const { error, httpStatus, timeout, aborted } = createErrorAPI<
  MockLanguageModel,
  'call'
>();
const { noImage } = createNoImageAPI<MockImageModel, 'call'>();

describe('error (call layer)', () => {
  it('should run the predicate against the current error', async () => {
    // Arrange
    const boom = new Error('boom');
    const seen: Array<unknown> = [];
    const cond = error<MockLanguageModel>((e) => {
      seen.push(e);
      return true;
    });

    // Act
    const matched = await cond.evaluate(buildCallErrorContext(boom));

    // Assert
    expect(matched).toBe(true);
    expect(seen[0]).toBe(boom);
  });

  it('should return false on result attempts', async () => {
    // Arrange
    const cond = error<MockLanguageModel>(() => true);

    // Act
    const matched = await cond.evaluate(
      buildCallResultContext(await callGenerateTextResult()),
    );

    // Assert
    expect(matched).toBe(false);
  });

  it('should pass the call context as the second argument', async () => {
    // Arrange
    const seen: Array<unknown> = [];
    const cond = error<MockLanguageModel>((_e, ctx) => {
      seen.push(ctx.current.type, ctx.attempts.length);
      return true;
    });

    // Act
    await cond.evaluate(buildCallErrorContext(new Error('boom')));

    // Assert
    expect(seen).toEqual(['error', 1]);
  });

  it('should match by error class', async () => {
    // Arrange
    const cond = error.isInstance<MockLanguageModel>(APICallError);

    // Act
    const matched = await cond.evaluate(
      buildCallErrorContext(apiError({ statusCode: 429 })),
    );
    const missed = await cond.evaluate(
      buildCallErrorContext(new Error('boom')),
    );

    // Assert
    expect(matched).toBe(true);
    expect(missed).toBe(false);
  });

  it('should match by the retryable flag', async () => {
    // Arrange
    const cond = error.isRetryable<MockLanguageModel>(true);

    // Act
    const matched = await cond.evaluate(
      buildCallErrorContext(Errors.rateLimited()),
    );

    // Assert
    expect(matched).toBe(true);
  });

  it('should match by status code', async () => {
    // Arrange
    const cond = error.statusCode<MockLanguageModel>(529);

    // Act
    const matched = await cond.evaluate(
      buildCallErrorContext(apiError({ statusCode: 529 })),
    );
    const missed = await cond.evaluate(
      buildCallErrorContext(apiError({ statusCode: 500 })),
    );

    // Assert
    expect(matched).toBe(true);
    expect(missed).toBe(false);
  });

  it('should match by message substring, case-insensitively', async () => {
    // Arrange
    const cond = error.message<MockLanguageModel>('OVERLOADED');

    // Act
    const matched = await cond.evaluate(
      buildCallErrorContext(new Error('service overloaded')),
    );

    // Assert
    expect(matched).toBe(true);
  });
});

describe('httpStatus (call layer)', () => {
  it('should match either the status code or the message', async () => {
    // Arrange
    const cond = httpStatus<MockLanguageModel>(529, 'overloaded');

    // Act
    const byCode = await cond.evaluate(
      buildCallErrorContext(apiError({ statusCode: 529 })),
    );
    const byMessage = await cond.evaluate(
      buildCallErrorContext(new Error('service overloaded')),
    );
    const missed = await cond.evaluate(
      buildCallErrorContext(new Error('unrelated')),
    );

    // Assert
    expect(byCode).toBe(true);
    expect(byMessage).toBe(true);
    expect(missed).toBe(false);
  });
});

describe('timeout / aborted (call layer)', () => {
  it('should tell a timeout apart from a manual abort', async () => {
    // Arrange
    const isTimeout = timeout<MockLanguageModel>();
    const isAborted = aborted<MockLanguageModel>();

    // Act
    const timedOut = buildCallErrorContext(timeoutError());
    const cancelled = buildCallErrorContext(abortError());

    // Assert
    expect(await isTimeout.evaluate(timedOut)).toBe(true);
    expect(await isTimeout.evaluate(cancelled)).toBe(false);
    expect(await isAborted.evaluate(cancelled)).toBe(true);
    expect(await isAborted.evaluate(timedOut)).toBe(false);
  });
});

describe('noImage (call layer)', () => {
  it('should match NoImageGeneratedError', async () => {
    // Arrange
    const cond = noImage<MockImageModel>();

    // Act
    const matched = await cond.evaluate(
      buildCallImageErrorContext(new NoImageGeneratedError({ responses: [] })),
    );
    const missed = await cond.evaluate(
      buildCallImageErrorContext(new Error('boom')),
    );

    // Assert
    expect(matched).toBe(true);
    expect(missed).toBe(false);
  });
});
