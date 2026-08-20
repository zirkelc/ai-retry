import { describe, expect, it } from 'vitest';
import { calculateExponentialBackoff } from './calculate-exponential-backoff.js';
 
describe('calculateExponentialBackoff', () => {
  it('returns base delay for 0 attempts', () => {
    expect(calculateExponentialBackoff(1000, 2, 0, undefined, false)).toBe(1000);
  });
 
  it('returns fixed delay for backoff factor 1', () => {
    expect(calculateExponentialBackoff(1000, 1, 1, undefined, false)).toBe(1000);
    expect(calculateExponentialBackoff(1000, 1, 2, undefined, false)).toBe(1000);
    expect(calculateExponentialBackoff(1000, 1, 3, undefined, false)).toBe(1000);
  });
 
  it('calculates exponential backoff correctly', () => {
    expect(calculateExponentialBackoff(1000, 2, 1, undefined, false)).toBe(2000);
    expect(calculateExponentialBackoff(1000, 2, 2, undefined, false)).toBe(4000);
    expect(calculateExponentialBackoff(1000, 2, 3, undefined, false)).toBe(8000);
    expect(calculateExponentialBackoff(500, 2, 2, undefined, false)).toBe(2000);
    expect(calculateExponentialBackoff(2000, 2, 2, undefined, false)).toBe(8000);
    expect(calculateExponentialBackoff(1000, 3, 2, undefined, false)).toBe(9000);
    expect(calculateExponentialBackoff(1000, 1.5, 2, undefined, false)).toBe(2250);
  });
 
  it('defaults backoff factor to 1 when not provided', () => {
    expect(calculateExponentialBackoff(1000, undefined, 1, undefined, false)).toBe(1000);
    expect(calculateExponentialBackoff(1000, undefined, 2, undefined, false)).toBe(1000);
    expect(calculateExponentialBackoff(1000, undefined, 3, undefined, false)).toBe(1000);
  });
 
  it('ensures backoff factor is at least 1', () => {
    expect(calculateExponentialBackoff(1000, 0, 2, undefined, false)).toBe(1000);
    expect(calculateExponentialBackoff(1000, -1, 2, undefined, false)).toBe(1000);
    expect(calculateExponentialBackoff(1000, 0.5, 2, undefined, false)).toBe(1000);
  });
 
  it('caps delay at maxDelay', () => {
    expect(calculateExponentialBackoff(1000, 2, 10, 5000, false)).toBe(5000);
    expect(calculateExponentialBackoff(1000, 2, 1, 500, false)).toBe(500);
  });
 
  it('applies jitter by default', () => {
    const delay1 = calculateExponentialBackoff(1000, 2, 2);
    const delay2 = calculateExponentialBackoff(1000, 2, 2);
    expect(delay1).not.toBe(delay2);
    expect(delay1).toBeLessThanOrEqual(4000);
    expect(delay2).toBeLessThanOrEqual(4000);
  });
 
  it('returns exact delay when jitter is disabled', () => {
    expect(calculateExponentialBackoff(1000, 2, 2, undefined, false)).toBe(4000);
  });
});
