import type { embedMany } from 'ai';

/**
 * What a `retryableEmbedMany` result condition judges: the completed
 * embeddings, exactly as the caller receives them.
 */
export type EmbedManyCommitResult = Awaited<ReturnType<typeof embedMany>>;
