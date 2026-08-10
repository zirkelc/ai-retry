import type { embed } from 'ai';

/**
 * What a `retryableEmbed` result condition judges: the completed embedding,
 * exactly as the caller receives it. A single value, so the embedding is
 * `embedding` rather than `embeddings`.
 */
export type EmbedCommitResult = Awaited<ReturnType<typeof embed>>;
