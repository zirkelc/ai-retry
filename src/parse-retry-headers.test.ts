import { describe, expect, it } from 'vitest';
import { parseRetryHeaders } from './parse-retry-headers.js';

describe('parseRetryHeaders', () => {
  describe('retry-after-ms header', () => {
    it('should parse valid retry-after-ms in milliseconds', () => {
      const headers = { 'retry-after-ms': '5000' };
      expect(parseRetryHeaders(headers)).toBe(5000);
    });

    it('should parse retry-after-ms with decimal values', () => {
      const headers = { 'retry-after-ms': '1234.56' };
      expect(parseRetryHeaders(headers)).toBe(1234.56);
    });

    it('should return uncapped retry-after-ms values', () => {
      const headers = { 'retry-after-ms': '120000' };
      expect(parseRetryHeaders(headers)).toBe(120000);
    });

    it('should return null for negative retry-after-ms', () => {
      const headers = { 'retry-after-ms': '-1000' };
      expect(parseRetryHeaders(headers)).toBeNull();
    });

    it('should return null for invalid retry-after-ms', () => {
      const headers = { 'retry-after-ms': 'invalid' };
      expect(parseRetryHeaders(headers)).toBeNull();
    });

    it('should prioritize retry-after-ms over retry-after', () => {
      const headers = {
        'retry-after-ms': '3000',
        'retry-after': '10',
      };
      expect(parseRetryHeaders(headers)).toBe(3000);
    });
  });

  describe('retry-after header (seconds)', () => {
    it('should parse retry-after in seconds', () => {
      const headers = { 'retry-after': '5' };
      expect(parseRetryHeaders(headers)).toBe(5000);
    });

    it('should parse retry-after with decimal seconds', () => {
      const headers = { 'retry-after': '2.5' };
      expect(parseRetryHeaders(headers)).toBe(2500);
    });

    it('should return uncapped retry-after values', () => {
      const headers = { 'retry-after': '120' };
      expect(parseRetryHeaders(headers)).toBe(120000);
    });

    it('should handle zero seconds', () => {
      const headers = { 'retry-after': '0' };
      expect(parseRetryHeaders(headers)).toBe(0);
    });
  });

  describe('retry-after header (HTTP date)', () => {
    it('should parse retry-after as HTTP date', () => {
      const futureDate = new Date(Date.now() + 5000);
      const headers = { 'retry-after': futureDate.toUTCString() };
      const delay = parseRetryHeaders(headers);

      expect(delay).not.toBeNull();
      expect(delay).toBeGreaterThanOrEqual(4000);
      expect(delay).toBeLessThanOrEqual(5000);
    });

    it('should return uncapped HTTP date delay', () => {
      const futureDate = new Date(Date.now() + 120000);
      const headers = { 'retry-after': futureDate.toUTCString() };
      const delay = parseRetryHeaders(headers);

      expect(delay).not.toBeNull();
      expect(delay).toBeGreaterThanOrEqual(119000);
      expect(delay).toBeLessThanOrEqual(120000);
    });

    it('should handle past dates as zero delay', () => {
      const pastDate = new Date(Date.now() - 5000);
      const headers = { 'retry-after': pastDate.toUTCString() };
      expect(parseRetryHeaders(headers)).toBe(0);
    });

    it('should return null for invalid date format', () => {
      const headers = { 'retry-after': 'not-a-valid-date' };
      expect(parseRetryHeaders(headers)).toBeNull();
    });
  });

  describe('edge cases', () => {
    it('should return null when headers are undefined', () => {
      expect(parseRetryHeaders(undefined)).toBeNull();
    });

    it('should return null when headers object is empty', () => {
      expect(parseRetryHeaders({})).toBeNull();
    });

    it('should return null when neither header is present', () => {
      const headers = { 'content-type': 'application/json' };
      expect(parseRetryHeaders(headers)).toBeNull();
    });

    it('should handle empty string values', () => {
      const headers = { 'retry-after': '' };
      expect(parseRetryHeaders(headers)).toBeNull();
    });

    it('should handle headers with only whitespace', () => {
      const headers = { 'retry-after': '   ' };
      expect(parseRetryHeaders(headers)).toBeNull();
    });
  });
});
