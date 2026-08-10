import type { generateImage } from 'ai';

/**
 * What a `retryableGenerateImage` result condition judges: the completed
 * generation, exactly as the caller receives it.
 */
export type GenerateImageCommitResult = Awaited<
  ReturnType<typeof generateImage>
>;
