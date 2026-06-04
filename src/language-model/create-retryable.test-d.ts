import { assertType, describe, expectTypeOf, it } from 'vitest';
import { MockLanguageModel } from '../internal/test-utils.js';
import type {
  LanguageModel,
  LanguageModelCallOptions,
  LanguageModelGenerate,
  LanguageModelStream,
  SuccessContext,
} from '../types.js';
import { createRetryableModel } from './create-retryable.js';

describe('createRetryableModel', () => {
  it('should return LanguageModel for a model instance', () => {
    const retryable = createRetryableModel({
      model: new MockLanguageModel(),
      retries: [new MockLanguageModel(), { model: new MockLanguageModel() }],
    });

    assertType<LanguageModel>(retryable);
    expectTypeOf(retryable).toEqualTypeOf<LanguageModel>();
  });

  it('should return LanguageModel for a gateway string', () => {
    const retryable = createRetryableModel({
      model: 'openai/gpt-4.1',
      retries: [
        'anthropic/claude-sonnet-4',
        { model: 'anthropic/claude-sonnet-4' },
      ],
    });

    assertType<LanguageModel>(retryable);
    expectTypeOf(retryable).toEqualTypeOf<LanguageModel>();
  });

  it('should type SuccessContext correctly', () => {
    type Ctx = SuccessContext<LanguageModel>;

    expectTypeOf<Ctx['current']['model']>().toEqualTypeOf<LanguageModel>();
    expectTypeOf<Ctx['current']['result']>().toEqualTypeOf<
      LanguageModelGenerate | LanguageModelStream
    >();
    expectTypeOf<
      Ctx['current']['options']
    >().toEqualTypeOf<LanguageModelCallOptions>();
    expectTypeOf<Ctx['current']['type']>().toEqualTypeOf<'success'>();
  });
});
