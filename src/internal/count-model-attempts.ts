import { getModelKey } from './get-model-key.js';
import type { CallRetryAttempt } from '../call/types.js';
import type { AnyModel, ModelRetryAttempt } from '../types.js';

/**
 * Count how many of the given attempts ran against the given model.
 *
 * Accepts either layer's attempt type — only each attempt's model is read. A
 * union of the two rather than a structurally loosened `{ model }`, so `MODEL`
 * keeps its meaning. A bare `{ model: MODEL }` cannot work: a result attempt's
 * model is `LanguageModel` outright rather than generic, so it is never
 * assignable to an unresolved `MODEL`.
 */
export function countModelAttempts<MODEL extends AnyModel>(
  model: MODEL,
  attempts: ReadonlyArray<ModelRetryAttempt<MODEL> | CallRetryAttempt<MODEL>>,
): number {
  const modelKey = getModelKey(model);
  return attempts.filter((a) => getModelKey(a.model) === modelKey).length;
}
