/**
 * `embed` with call-level retries.
 *
 *   import { retryableEmbed } from 'ai-retry/embed';
 *
 * The conditions that go in its `retry` list live one level down, at
 * `ai-retry/embed/conditions`.
 */

export { retryableEmbed, type RetryableEmbed } from './embed.js';
export type { EmbedCommitResult } from './types.js';
export type { EmbedInput } from '../inputs.js';
