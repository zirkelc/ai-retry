import type { LanguageModelV2 } from '@ai-sdk/provider';

export type LanguageModelV2Generate = Awaited<
  ReturnType<LanguageModelV2['doGenerate']>
>;

export type LanguageModelV2Stream = Awaited<
  ReturnType<LanguageModelV2['doStream']>
>;
