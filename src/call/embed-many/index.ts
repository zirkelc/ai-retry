/**
 * `embedMany` with call-level retries.
 *
 *   import { retryableEmbedMany } from 'ai-retry/embed-many';
 *
 * The conditions that go in its `retry` list live one level down, at
 * `ai-retry/embed-many/conditions`.
 */

export { retryableEmbedMany, type RetryableEmbedMany } from './embed-many.js';
export type { EmbedManyCommitResult } from './types.js';
export type { EmbedManyInput } from '../inputs.js';
