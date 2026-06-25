import { expect } from 'vitest';

/**
 * AI SDK v7 adds a `performance` object to stream `finish-step` parts holding
 * wall-clock timings (responseTimeMs, tokens-per-second, chunk gap stats, ...).
 * Those values are non-deterministic, so strip the whole object before it
 * reaches a snapshot to keep stream snapshots stable across runs.
 */
expect.addSnapshotSerializer({
  test: (val) =>
    val != null &&
    typeof val === 'object' &&
    !Array.isArray(val) &&
    'performance' in val &&
    typeof (val as Record<string, unknown>).performance === 'object',
  serialize: (val, config, indentation, depth, refs, printer) => {
    const { performance: _performance, ...rest } = val as Record<
      string,
      unknown
    >;
    return printer(rest, config, indentation, depth, refs);
  },
});
