import { describe, expect, it } from 'vitest';
import { mergeRetryTimeout, totalTimeoutMs } from './retry-timeout.js';

describe('totalTimeoutMs', () => {
  it('should read a number as the total budget', () => {
    // Assert
    expect(totalTimeoutMs(5_000)).toBe(5_000);
  });

  it('should read the total out of the object form', () => {
    // Assert
    expect(totalTimeoutMs({ totalMs: 5_000, stepMs: 1_000 })).toBe(5_000);
  });

  it('should report no budget when only finer windows are named', () => {
    // Arrange — nothing here bounds the call as a whole, so there is no
    // deadline a signal could carry.
    expect(totalTimeoutMs({ chunkMs: 100, firstChunkMs: 200 })).toBeUndefined();
  });

  it('should report no budget when there is no deadline at all', () => {
    // Assert
    expect(totalTimeoutMs(undefined)).toBeUndefined();
  });
});

describe('mergeRetryTimeout', () => {
  it('should keep the windows the call configured', () => {
    // Arrange — the retry narrows the total only.
    const base = { totalMs: 30_000, firstChunkMs: 2_000, chunkMs: 500 };

    // Act
    const merged = mergeRetryTimeout(base, { totalMs: 5_000 });

    // Assert
    expect(merged).toEqual({
      totalMs: 5_000,
      firstChunkMs: 2_000,
      chunkMs: 500,
    });
  });

  it('should expand a number on either side to a total', () => {
    // Act
    const overNumber = mergeRetryTimeout(10_000, { stepMs: 1_000 });
    const asNumber = mergeRetryTimeout({ chunkMs: 500 }, 5_000);

    // Assert
    expect(overNumber).toEqual({ totalMs: 10_000, stepMs: 1_000 });
    expect(asNumber).toEqual({ chunkMs: 500, totalMs: 5_000 });
  });

  it('should merge per-tool budgets rather than replacing them', () => {
    // Arrange
    const base = { tools: { searchMs: 1_000, fetchMs: 2_000 } };

    // Act
    const merged = mergeRetryTimeout(base, { tools: { searchMs: 500 } });

    // Assert — the retry narrows one tool and leaves the other alone.
    expect(merged).toEqual({ tools: { searchMs: 500, fetchMs: 2_000 } });
  });

  it('should not let an absent window erase one the call set', () => {
    // Arrange — spelling a key as `undefined` is not the same as narrowing it.
    const base = { totalMs: 30_000, stepMs: 1_000 };

    // Act
    const merged = mergeRetryTimeout(base, { totalMs: undefined, stepMs: 500 });

    // Assert
    expect(merged).toEqual({ totalMs: 30_000, stepMs: 500 });
  });

  it('should stand alone when the call configured nothing', () => {
    // Act
    const merged = mergeRetryTimeout(undefined, { totalMs: 5_000 });

    // Assert
    expect(merged).toEqual({ totalMs: 5_000 });
  });
});
