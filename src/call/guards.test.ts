import { describe, expect, it } from 'vitest';
import {
  callEmbedManyResult,
  callEmbedResult,
  callGenerateTextResult,
  callImageResult,
  callStreamTextResult,
} from '../internal/test-utils.js';
import {
  isEmbedManyResult,
  isEmbedResult,
  isGenerateImageResult,
  isGenerateTextResult,
  isStreamTextResult,
} from './guards.js';

describe('call-layer result guards', () => {
  describe('isGenerateTextResult', () => {
    it('should match a generateText result', async () => {
      // Arrange
      const result = await callGenerateTextResult();

      // Act
      const matched = isGenerateTextResult(result);

      // Assert
      expect(matched).toBe(true);
    });

    it('should not match the other member of its family', () => {
      // Arrange
      const result = callStreamTextResult();

      // Act
      const matched = isGenerateTextResult(result);

      // Assert
      expect(matched).toBe(false);
    });
  });

  describe('isStreamTextResult', () => {
    it('should match a contentless streamText result', () => {
      // Arrange
      const result = callStreamTextResult('content-filter');

      // Act & Assert
      expect(isStreamTextResult(result)).toBe(true);
    });

    it('should not match a generateText result', async () => {
      // Arrange
      const result = await callGenerateTextResult();

      // Act & Assert
      expect(isStreamTextResult(result)).toBe(false);
    });
  });

  describe('isEmbedResult / isEmbedManyResult', () => {
    it('should tell the two embedding entry points apart', async () => {
      // Arrange
      const one = await callEmbedResult();
      const many = await callEmbedManyResult();

      // Act & Assert
      expect(isEmbedResult(one)).toBe(true);
      expect(isEmbedManyResult(one)).toBe(false);
      expect(isEmbedResult(many)).toBe(false);
      expect(isEmbedManyResult(many)).toBe(true);
    });
  });

  describe('isGenerateImageResult', () => {
    it('should match the only member of its family', async () => {
      // Arrange
      const result = await callImageResult();

      // Act & Assert
      expect(isGenerateImageResult(result)).toBe(true);
    });
  });
});
