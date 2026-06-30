import { generateText } from 'ai';
import { describe, expect, it } from 'vitest';
import {
  Errors,
  MockLanguageModel,
  buildErrorContext,
  createRetryableModel,
} from '../test-utils.js';
import { createErrorAPI } from './error.js';

const { httpStatus } = createErrorAPI<MockLanguageModel>();

describe('httpStatus', () => {
  it(`should match by numeric status code`, async () => {
    // Arrange
    const cond = httpStatus<MockLanguageModel>(529);

    // Act
    const matched = await cond.evaluate(
      buildErrorContext(Errors.from({ statusCode: 529 })),
    );
    const missed = await cond.evaluate(
      buildErrorContext(Errors.from({ statusCode: 503 })),
    );

    // Assert
    expect(matched).toBe(true);
    expect(missed).toBe(false);
  });

  it(`should match by message substring`, async () => {
    // Arrange
    const cond = httpStatus<MockLanguageModel>('overloaded');

    // Act
    const matched = await cond.evaluate(
      buildErrorContext(Errors.from({ message: 'Service overloaded' })),
    );
    const missed = await cond.evaluate(
      buildErrorContext(Errors.from({ message: 'Bad gateway' })),
    );

    // Assert
    expect(matched).toBe(true);
    expect(missed).toBe(false);
  });

  it(`should match by message regex`, async () => {
    // Arrange
    const cond = httpStatus<MockLanguageModel>(/rate.?limit/i);

    // Act
    const matched = await cond.evaluate(
      buildErrorContext(
        Errors.from({ statusCode: 429, message: 'Rate Limit hit' }),
      ),
    );
    const missed = await cond.evaluate(
      buildErrorContext(
        Errors.from({ statusCode: 429, message: 'Too many requests' }),
      ),
    );

    // Assert
    expect(matched).toBe(true);
    expect(missed).toBe(false);
  });

  it(`should match by status-code regex`, async () => {
    // Arrange
    const cond = httpStatus<MockLanguageModel>(/^5\d\d$/);

    // Act
    const matched500 = await cond.evaluate(
      buildErrorContext(Errors.from({ statusCode: 500, message: 'no match' })),
    );
    const matched599 = await cond.evaluate(
      buildErrorContext(Errors.from({ statusCode: 599, message: 'no match' })),
    );
    const missed = await cond.evaluate(
      buildErrorContext(Errors.from({ statusCode: 404, message: 'no match' })),
    );

    // Assert
    expect(matched500).toBe(true);
    expect(matched599).toBe(true);
    expect(missed).toBe(false);
  });

  it(`should accept a mix of status, string, and regex`, async () => {
    // Arrange
    const cond = httpStatus<MockLanguageModel>(
      529,
      'overloaded',
      /rate.?limit/i,
    );

    // Act
    const byStatus = await cond.evaluate(
      buildErrorContext(Errors.from({ statusCode: 529 })),
    );
    const byString = await cond.evaluate(
      buildErrorContext(Errors.from({ message: 'overloaded' })),
    );
    const byRegex = await cond.evaluate(
      buildErrorContext(
        Errors.from({ statusCode: 429, message: 'Rate-Limit reached' }),
      ),
    );
    const noMatch = await cond.evaluate(
      buildErrorContext(
        Errors.from({ statusCode: 401, message: 'Unauthorized' }),
      ),
    );

    // Assert
    expect(byStatus).toBe(true);
    expect(byString).toBe(true);
    expect(byRegex).toBe(true);
    expect(noMatch).toBe(false);
  });

  it(`should not match plain errors`, async () => {
    // Arrange
    const cond = httpStatus<MockLanguageModel>(500);

    // Act
    const matched = await cond.evaluate(buildErrorContext(new Error('plain')));

    // Assert
    expect(matched).toBe(false);
  });

  it(`should switch to fallback in createRetryableModel`, async () => {
    // Arrange
    const baseModel = MockLanguageModel.from({
      doGenerate: Errors.from({ statusCode: 529, message: 'overloaded' }),
    });
    const fallback = MockLanguageModel.from('hello');

    // Act
    const result = await generateText({
      model: createRetryableModel({
        model: baseModel,
        retries: [
          httpStatus<MockLanguageModel>(529, 'overloaded').switch({
            model: fallback,
          }),
        ],
      }),
      prompt: 'hi',
      maxRetries: 0,
    });

    // Assert
    expect(baseModel.doGenerate).toHaveBeenCalledTimes(1);
    expect(fallback.doGenerate).toHaveBeenCalledTimes(1);
    expect(result.text).toBe('hello');
  });
});
