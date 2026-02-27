import { getModelKey } from './get-model-key.js';
import type {
  EmbeddingModel,
  ImageModel,
  LanguageModel,
  RetryAttempt,
} from './types.js';

export function countModelAttempts<
  MODEL extends LanguageModel | EmbeddingModel | ImageModel,
>(model: MODEL, attempts: ReadonlyArray<RetryAttempt<MODEL>>): number {
  const modelKey = getModelKey(model);
  return attempts.filter((a) => getModelKey(a.model) === modelKey).length;
}
