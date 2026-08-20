import { assertType, describe, expectTypeOf, it } from 'vitest';
import { MockLanguageModel } from '../../internal/test-utils.js';
import type {
  LanguageModel,
  LanguageModelCallOptions,
  LanguageModelGenerate,
  LanguageModelStream,
  ModelSuccessContext,
} from '../../types.js';
import { timeout } from './conditions/index.js';
import { createRetryableModel } from './create-retryable-model.js';

describe('createRetryableModel', () => {
  it('should return LanguageModel for a model instance', () => {
    const retryable = createRetryableModel({
      model: MockLanguageModel.from(),
      retries: [MockLanguageModel.from(), { model: MockLanguageModel.from() }],
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

  it('should type ModelSuccessContext correctly', () => {
    type Ctx = ModelSuccessContext<LanguageModel>;

    expectTypeOf<Ctx['current']['model']>().toEqualTypeOf<LanguageModel>();
    expectTypeOf<Ctx['current']['result']>().toEqualTypeOf<
      LanguageModelGenerate | LanguageModelStream
    >();
    expectTypeOf<
      Ctx['current']['options']
    >().toEqualTypeOf<LanguageModelCallOptions>();
    expectTypeOf<Ctx['current']['type']>().toEqualTypeOf<'success'>();
  });

  it('should take a retry deadline only as a number', () => {
    // Arrange — a retryable model applies its deadline by building an
    // `AbortSignal`, which can carry a wall-clock budget and nothing else. The
    // SDK's structured windows belong to the call layer, which has a real
    // `timeout` argument to put them in.
    createRetryableModel({
      model: MockLanguageModel.from(),
      retries: [{ model: MockLanguageModel.from(), timeout: 5_000 }],
    });

    createRetryableModel({
      model: MockLanguageModel.from(),
      retries: [
        // @ts-expect-error a structured deadline cannot be expressed as a signal
        { model: MockLanguageModel.from(), timeout: { totalMs: 5_000 } },
      ],
    });
  });

  it('should reject a structured deadline from a condition too', () => {
    // Arrange — hoisted, so the deadline is inferred from the target rather
    // than from the list it is about to land in. That is what makes the next
    // assertion about the list rather than about the `switch` call.
    const numeric = timeout().switch({
      model: MockLanguageModel.from(),
      timeout: 5_000,
    });
    const structured = timeout().switch({
      model: MockLanguageModel.from(),
      timeout: { totalMs: 5_000 },
    });

    // Assert
    createRetryableModel({
      model: MockLanguageModel.from(),
      retries: [numeric],
    });

    createRetryableModel({
      model: MockLanguageModel.from(),
      // @ts-expect-error same rule, reported where the retryable lands
      retries: [structured],
    });
  });
});
