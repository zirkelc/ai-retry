import { describe, expect, it } from 'vitest';
import { RefusalError, refusalGate } from './refusal-gate.js';

const REFUSAL = "I'm sorry, but I cannot assist";

describe('refusalGate', () => {
  it('should wait while the text is still a prefix of a phrase', () => {
    // Arrange
    const gate = refusalGate([REFUSAL]);

    // Act
    const decision = gate("I'm sorry, ");

    // Assert
    expect(decision).toBe('wait');
  });

  it('should throw a RefusalError once a full phrase is matched', () => {
    // Arrange
    const gate = refusalGate([REFUSAL]);

    // Act
    const act = () => gate("I'm sorry, but I cannot assist with that.");

    // Assert
    expect(act).toThrow(RefusalError);
  });

  it('should commit once the text diverges from every phrase', () => {
    // Arrange
    const gate = refusalGate([REFUSAL]);

    // Act
    const decision = gate("I'm sorry to hear that");

    // Assert
    expect(decision).toBe('commit');
  });

  it('should match case- and whitespace-insensitively', () => {
    // Arrange
    const gate = refusalGate([REFUSAL]);

    // Act
    const act = () => gate("I'M   SORRY,\n BUT I CANNOT ASSIST");

    // Assert
    expect(act).toThrow(RefusalError);
  });

  it('should carry the matched phrase and buffered text on the error', () => {
    // Arrange
    const gate = refusalGate([REFUSAL]);
    const buffered = "I'm sorry, but I cannot assist you.";

    // Act
    let caught: unknown;
    try {
      gate(buffered);
    } catch (e) {
      caught = e;
    }

    // Assert
    expect(caught).toBeInstanceOf(RefusalError);
    expect((caught as RefusalError).phrase).toBe(REFUSAL);
    expect((caught as RefusalError).bufferedText).toBe(buffered);
    expect((caught as RefusalError).name).toBe('RefusalError');
  });

  it('should throw a custom error from onRefusal', () => {
    // Arrange
    const gate = refusalGate([REFUSAL], {
      onRefusal: ({ phrase }) =>
        Object.assign(new Error(`blocked: ${phrase}`), {
          name: 'BlockedError',
        }),
    });

    // Act
    const act = () => gate("I'm sorry, but I cannot assist");

    // Assert
    expect(act).toThrow('blocked');
  });

  it('should wait on a prefix of any of several phrases', () => {
    // Arrange
    const gate = refusalGate([REFUSAL, 'I cannot help with that']);

    // Act
    const decision = gate('I cannot help');

    // Assert
    expect(decision).toBe('wait');
  });

  it('should ignore empty phrases', () => {
    // Arrange
    const gate = refusalGate(['', '   ', REFUSAL]);

    // Act
    const decision = gate('A real answer.');

    // Assert
    expect(decision).toBe('commit');
  });
});
