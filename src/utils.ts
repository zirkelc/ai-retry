import type {
  LanguageModelV2Generate,
  LanguageModelV2Stream,
} from './types.js';

export const isObject = (value: unknown): value is Record<string, unknown> =>
  typeof value === 'object' && value !== null;

export const isString = (value: unknown): value is string =>
  typeof value === 'string';

export const isStreamResult = (
  result: LanguageModelV2Generate | LanguageModelV2Stream,
): result is LanguageModelV2Stream => 'stream' in result;

export const isGenerateResult = (
  result: LanguageModelV2Generate | LanguageModelV2Stream,
): result is LanguageModelV2Generate => 'content' in result;
