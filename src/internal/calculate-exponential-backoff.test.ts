import { describe, expect, it } from 'vitest';
import { calculateExponentialBackoff } from './calculate-exponential-backoff.js';

describe('calculateExponentialBackoff', () => {
  it('returns base delay for 0 attempts', () => {
    expect(calculateExponentialBackoff(1000, 2, 0)).toBe(1000);
  });

  it('returns fixed delay for backoff factor 1', () => {
    expect(calculateExponentialBackoff(1000, 1, 1)).toBe(1000);
    expect(calculateExponentialBackoff(1000, 1, 2)).toBe(1000);
    expect(calculateExponentialBackoff(1000, 1, 3)).toBe(1000);
  });

  it('calculates exponential backoff correctly', () => {
    expect(calculateExponentialBackoff(1000, 2, 1)).toBe(2000);
    expect(calculateExponentialBackoff(1000, 2, 2)).toBe(4000);
    expect(calculateExponentialBackoff(1000, 2, 3)).toBe(8000);
    expect(calculateExponentialBackoff(500, 2, 2)).toBe(2000);
    expect(calculateExponentialBackoff(2000, 2, 2)).toBe(8000);
    expect(calculateExponentialBackoff(1000, 3, 2)).toBe(9000);
    expect(calculateExponentialBackoff(1000, 1.5, 2)).toBe(2250);
  });

  it('defaults backoff factor to 1 when not provided', () => {
    expect(calculateExponentialBackoff(1000, undefined, 1)).toBe(1000);
    expect(calculateExponentialBackoff(1000, undefined, 2)).toBe(1000);
    expect(calculateExponentialBackoff(1000, undefined, 3)).toBe(1000);
  });

  it('ensures backoff factor is at least 1', () => {
    expect(calculateExponentialBackoff(1000, 0, 2)).toBe(1000);
    expect(calculateExponentialBackoff(1000, -1, 2)).toBe(1000);
    expect(calculateExponentialBackoff(1000, 0.5, 2)).toBe(1000);
  });
});
