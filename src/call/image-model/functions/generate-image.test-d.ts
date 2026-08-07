import { generateImage } from 'ai';
import { describe, expectTypeOf, it } from 'vitest';
import {
  MockImageModel,
  MockLanguageModel,
} from '../../../internal/test-utils.js';
import { retryableGenerateImage } from './generate-image.js';

const imageModel = MockImageModel.from();

describe('retryableGenerateImage', () => {
  it('should keep the result type identical to a direct call', async () => {
    // Act
    const direct = await generateImage({ model: imageModel, prompt: 'a cat' });
    const wrapped = await retryableGenerateImage({
      model: imageModel,
      prompt: 'a cat',
      n: 2,
      size: '512x512',
      retry: [MockImageModel.from()],
    });

    // Assert
    expectTypeOf(wrapped.image).toEqualTypeOf<typeof direct.image>();
    expectTypeOf(wrapped.images).toEqualTypeOf<typeof direct.images>();
  });

  it('should reject a fallback from the wrong model family', () => {
    // Assert
    retryableGenerateImage({
      model: imageModel,
      prompt: 'a cat',
      // @ts-expect-error a language model is not an image fallback
      retry: [MockLanguageModel.from()],
    });
  });

  it('should accept the overrides it actually takes', () => {
    // Assert
    retryableGenerateImage({
      model: imageModel,
      prompt: 'a cat',
      retry: [{ model: imageModel, options: { prompt: 'a dog', n: 2 } }],
    });
  });

  it('should accept the bare-array shorthand', async () => {
    // Act
    const direct = await generateImage({ model: imageModel, prompt: 'a cat' });
    const wrapped = await retryableGenerateImage({
      model: imageModel,
      prompt: 'a cat',
      retry: [MockImageModel.from()],
    });

    // Assert — the shorthand does not disturb the entry point's own inference.
    expectTypeOf(wrapped.images).toEqualTypeOf<typeof direct.images>();
  });

  it('should accept the object form with hooks', () => {
    // Assert
    retryableGenerateImage({
      model: imageModel,
      prompt: 'a cat',
      retry: {
        retries: [MockImageModel.from()],
        disabled: false,
        onError: () => {},
        onRetry: () => {},
        onFailure: () => {},
      },
    });
  });

  it('should type onSuccess with the entry point result', async () => {
    // Act
    const direct = await generateImage({ model: imageModel, prompt: 'a cat' });

    // Assert — the hook sees the same result the caller does.
    await retryableGenerateImage({
      model: imageModel,
      prompt: 'a cat',
      retry: {
        retries: [],
        onSuccess: (context) => {
          expectTypeOf(context.current.result.images).toEqualTypeOf<
            typeof direct.images
          >();
        },
      },
    });
  });
});
