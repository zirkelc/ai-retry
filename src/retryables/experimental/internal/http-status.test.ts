import { generateText } from 'ai';
import { describe, expect, it } from 'vitest';
import { createRetryable } from '../../../create-retryable-model.js';
import {
  apiError,
  buildErrorContext,
  generateTextResult,
  MockLanguageModel,
} from '../../../test-utils.js';
import { createErrorAPI } from './create-error-api.js';

const { httpStatus } = createErrorAPI<MockLanguageModel>();

describe('httpStatus', () => {
  it(`should match by numeric status code`, async () => {
    // Arrange
    const cond = httpStatus<MockLanguageModel>(529);

    // Act
    const matched = await cond.evaluate(
      buildErrorContext(apiError({ statusCode: 529 })),
    );
    const missed = await cond.evaluate(
      buildErrorContext(apiError({ statusCode: 503 })),
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
      buildErrorContext(apiError({ message: 'Service overloaded' })),
    );
    const missed = await cond.evaluate(
      buildErrorContext(apiError({ message: 'Bad gateway' })),
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
        apiError({ statusCode: 429, message: 'Rate Limit hit' }),
      ),
    );
    const missed = await cond.evaluate(
      buildErrorContext(
        apiError({ statusCode: 429, message: 'Too many requests' }),
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
      buildErrorContext(apiError({ statusCode: 500, message: 'no match' })),
    );
    const matched599 = await cond.evaluate(
      buildErrorContext(apiError({ statusCode: 599, message: 'no match' })),
    );
    const missed = await cond.evaluate(
      buildErrorContext(apiError({ statusCode: 404, message: 'no match' })),
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
      buildErrorContext(apiError({ statusCode: 529 })),
    );
    const byString = await cond.evaluate(
      buildErrorContext(apiError({ message: 'overloaded' })),
    );
    const byRegex = await cond.evaluate(
      buildErrorContext(
        apiError({ statusCode: 429, message: 'Rate-Limit reached' }),
      ),
    );
    const noMatch = await cond.evaluate(
      buildErrorContext(apiError({ statusCode: 401, message: 'Unauthorized' })),
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

  it(`should switch to fallback in createRetryable`, async () => {
    // Arrange
    const baseModel = new MockLanguageModel({
      doGenerate: apiError({ statusCode: 529, message: 'overloaded' }),
    });
    const fallback = new MockLanguageModel({
      doGenerate: generateTextResult('hello'),
    });

    // Act
    const result = await generateText({
      model: createRetryable({
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
