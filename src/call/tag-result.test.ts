import { embed, generateText } from 'ai';
import { describe, expect, it } from 'vitest';
import {
  Embedding,
  MockEmbeddingModel,
  MockLanguageModel,
} from '../internal/test-utils.js';
import { tagResult } from './tag-result.js';

const text = 'Hello, world!';

/**
 * A `generateText` result, which the SDK returns as a class instance whose
 * interesting fields are prototype getters.
 */
const generateTextResult = () =>
  generateText({ model: MockLanguageModel.from(text), prompt: 'Hello!' });

/** An `embed` result, which the SDK returns as a plain object. */
const embedResult = () =>
  embed({
    model: MockEmbeddingModel.from([Embedding.vector(3)]),
    value: 'Hello!',
  });

describe('tagResult', () => {
  it('should expose the operation tag', async () => {
    // Arrange
    const result = await generateTextResult();

    // Act
    const tagged = tagResult('generateText', result);

    // Assert
    expect(tagged.operation).toBe('generateText');
  });

  it('should preserve fields the SDK exposes as prototype getters', async () => {
    // Arrange — the reason this is a view and not a copy: `{ ...result }`
    // yields an object whose `text` and `toolCalls` are `undefined`, because
    // neither is an own property.
    const result = await generateTextResult();
    const ownKeys = Object.keys(result);

    // Act
    const tagged = tagResult('generateText', result);

    // Assert
    expect(ownKeys.includes('text')).toBe(false);
    expect(tagged.text).toBe(text);
    expect(tagged.finishReason).toBe('stop');
    expect(tagged.toolCalls).toEqual([]);
  });

  it('should preserve own enumerable properties', async () => {
    // Arrange
    const result = await embedResult();

    // Act
    const tagged = tagResult('embed', result);

    // Assert
    expect(tagged.embedding).toEqual(result.embedding);
    expect(tagged.value).toBe('Hello!');
  });

  it('should report the tag through the `in` operator', async () => {
    // Arrange
    const result = await embedResult();

    // Act
    const tagged = tagResult('embed', result);

    // Assert
    expect('operation' in tagged).toBe(true);
    expect('embedding' in tagged).toBe(true);
    expect('nope' in tagged).toBe(false);
  });

  it('should include the tag in Object.keys alongside the real keys', async () => {
    // Arrange
    const result = await embedResult();

    // Act
    const keys = Object.keys(tagResult('embed', result));

    // Assert
    expect(keys.includes('operation')).toBe(true);
    expect(keys.includes('embedding')).toBe(true);
  });

  it('should carry the tag through a spread', async () => {
    // Arrange
    const result = await embedResult();

    // Act
    const spread = { ...tagResult('embed', result) };

    // Assert — enumeration matches the underlying result, plus the tag.
    expect(spread.operation).toBe('embed');
    expect(spread.embedding).toEqual(result.embedding);
  });

  it('should not add the tag to the underlying result', async () => {
    // Arrange
    const result = await generateTextResult();

    // Act
    tagResult('generateText', result);

    // Assert — the object the caller receives is untouched.
    expect('operation' in result).toBe(false);
    expect(Object.keys(result).includes('operation')).toBe(false);
  });

  it('should keep each tag independent when one result is tagged twice', async () => {
    // Arrange
    const result = await embedResult();

    // Act
    const first = tagResult('embed', result);
    const second = tagResult('embedMany', result);

    // Assert
    expect(first.operation).toBe('embed');
    expect(second.operation).toBe('embedMany');
  });
});
