import { describe, expect, test } from 'vitest';
import { add, subtract } from '../src/math.js';

describe('math', () => {
  test('should add two numbers', () => {
    expect(add(1, 2)).toBe(3);
  });

  test('should subtract two numbers', () => {
    expect(subtract(1, 2)).toBe(-1);
  });
});
