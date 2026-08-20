import { describe, expect, it } from 'vitest';
import { countModelAttempts } from './count-model-attempts.js';
import { MockLanguageModel } from './test-utils.js';
import type { CallArgs, CallRetryAttempt } from '../call/types.js';
import type {
  LanguageModel,
  LanguageModelCallOptions,
  ModelRetryAttempt,
} from '../types.js';

describe('countModelAttempts', () => {
  const mockModel1 = MockLanguageModel.from();
  const mockModel2 = MockLanguageModel.from();
  const mockOptions: LanguageModelCallOptions = {
    prompt: [],
  };

  it('should return 0 when no attempts', () => {
    const attempts: Array<ModelRetryAttempt<LanguageModel>> = [];
    expect(countModelAttempts(mockModel1, attempts)).toBe(0);
  });

  it('should count single model attempts', () => {
    const attempts: Array<ModelRetryAttempt<LanguageModel>> = [
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
    const attempts: Array<ModelRetryAttempt<LanguageModel>> = [
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
    const attempts: Array<ModelRetryAttempt<LanguageModel>> = [
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

  it('should count call-layer attempts the same way', () => {
    // Arrange — the helper is shared, and a call attempt records the entry
    // point's arguments where a model attempt records provider call options.
    const attempts: Array<CallRetryAttempt<LanguageModel>> = [
      {
        type: 'error',
        error: new Error('test'),
        model: mockModel1,
        options: {} as CallArgs<LanguageModel>,
      },
      {
        type: 'error',
        error: new Error('test'),
        model: mockModel2,
        options: {} as CallArgs<LanguageModel>,
      },
    ];

    // Act & Assert
    expect(countModelAttempts(mockModel1, attempts)).toBe(1);
    expect(countModelAttempts(mockModel2, attempts)).toBe(1);
  });
});
