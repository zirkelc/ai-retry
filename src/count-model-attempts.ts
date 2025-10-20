import { getModelKey } from './get-model-key.js';
import type {
  EmbeddingModelV2,
  LanguageModelV2,
  RetryAttempt,
} from './types.js';

export function countModelAttempts<
  MODEL extends LanguageModelV2 | EmbeddingModelV2,
>(model: MODEL, attempts: ReadonlyArray<RetryAttempt<MODEL>>): number {
  const modelKey = getModelKey(model);
  return attempts.filter((a) => getModelKey(a.model) === modelKey).length;
}
