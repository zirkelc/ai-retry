import { NoImageGeneratedError } from 'ai';
import { describe, expect, it } from 'vitest';
import { buildImageErrorContext, MockImageModel } from '../../../test-utils.js';
import { noImage } from './no-image.js';

describe('noImage', () => {
  it(`should match NoImageGeneratedError`, async () => {
    // Arrange
    const cond = noImage<MockImageModel>();
    const err = new NoImageGeneratedError({ responses: [] });

    // Act
    const matched = await cond.evaluate(buildImageErrorContext(err));

    // Assert
    expect(matched).toBe(true);
  });

  it(`should not match a plain Error`, async () => {
    // Arrange
    const cond = noImage<MockImageModel>();

    // Act
    const matched = await cond.evaluate(
      buildImageErrorContext(new Error('boom')),
    );

    // Assert
    expect(matched).toBe(false);
  });
});
