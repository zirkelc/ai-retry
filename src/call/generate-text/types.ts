import type { generateText, ToolSet } from 'ai';

/**
 * What a `retryableGenerateText` result condition judges: the completed
 * generation, exactly as the caller receives it.
 *
 * A resolved `generateText` is a finished generation whose every field is
 * already settled, so there is nothing to read early and nothing spent by
 * reading it. The commit result is therefore the result itself.
 *
 * `TOOLS` is not inferred from the call — a condition is written against the
 * entry point, not against one call site, so there is nothing to infer it from.
 * It is whatever the condition names, and nothing checks that against the tools
 * the call was actually issued with; the contract is a cast's.
 */
export type GenerateTextCommitResult<TOOLS extends ToolSet = ToolSet> = Awaited<
  ReturnType<typeof generateText<TOOLS>>
>;
