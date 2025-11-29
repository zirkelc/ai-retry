import { describe, expect, it } from 'vitest';
import { countModelAttempts } from './count-model-attempts.js';
import { MockLanguageModel } from './test-utils.js';
import type {
  LanguageModel,
  LanguageModelRetryCallOptions,
  RetryAttempt,
} from './types.js';

describe('countModelAttempts', () => {
  const mockModel1 = new MockLanguageModel();
  const mockModel2 = new MockLanguageModel();
  const mockOptions: LanguageModelRetryCallOptions = {
    prompt: [],
  };

  it('should return 0 when no attempts', () => {
    const attempts: Array<RetryAttempt<LanguageModel>> = [];
    expect(countModelAttempts(mockModel1, attempts)).toBe(0);
  });

  it('should count single model attempts', () => {
    const attempts: Array<RetryAttempt<LanguageModel>> = [
      {
        type: 'error',
        error: new Error('test'),
        model: mockModel1,
        options: mockOptions,
      },
      {
        type: 'error',
        error: new Error('test'),
        model: mockModel1,
        options: mockOptions,
      },
      {
        type: 'error',
        error: new Error('test'),
        model: mockModel1,
        options: mockOptions,
      },
    ];
    expect(countModelAttempts(mockModel1, attempts)).toBe(3);
  });

  it('should count only matching model attempts', () => {
    const attempts: Array<RetryAttempt<LanguageModel>> = [
      {
        type: 'error',
        error: new Error('test'),
        model: mockModel1,
        options: mockOptions,
      },
      {
        type: 'error',
        error: new Error('test'),
        model: mockModel2,
        options: mockOptions,
      },
      {
        type: 'error',
        error: new Error('test'),
        model: mockModel1,
        options: mockOptions,
      },
      {
        type: 'error',
        error: new Error('test'),
        model: mockModel2,
        options: mockOptions,
      },
      {
        type: 'error',
        error: new Error('test'),
        model: mockModel1,
        options: mockOptions,
      },
    ];
    expect(countModelAttempts(mockModel1, attempts)).toBe(3);
    expect(countModelAttempts(mockModel2, attempts)).toBe(2);
  });

  it('should return 0 when no matching model', () => {
    const attempts: Array<RetryAttempt<LanguageModel>> = [
      {
        type: 'error',
        error: new Error('test'),
        model: mockModel2,
        options: mockOptions,
      },
      {
        type: 'error',
        error: new Error('test'),
        model: mockModel2,
        options: mockOptions,
      },
    ];
    expect(countModelAttempts(mockModel1, attempts)).toBe(0);
  });
});
