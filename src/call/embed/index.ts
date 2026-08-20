/**
 * `embed` with call-level retries.
 *
 *   import { experimental_retryableEmbed } from 'ai-retry/embed';
 *
 * The conditions that go in its `retry` list live one level down, at
 * `ai-retry/embed/conditions`.
 */

export {
  retryableEmbed as experimental_retryableEmbed,
  type RetryableEmbed,
} from './embed.js';
export type { EmbedCommitResult } from './types.js';
export type { EmbedInput } from '../inputs.js';
