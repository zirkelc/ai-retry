import type { ProviderMetadata } from 'ai';
import type { CallFinishReason, CallLanguageModelUsage } from '../types.js';

/**
 * What a `retryableStreamText` result condition judges: a stream that finished
 * without ever emitting content.
 *
 * Deliberately not the SDK's own `StreamTextResult`, which is what the caller
 * receives. Every field on that is a promise that settles only once the stream
 * has been consumed, and consuming it is precisely what a pre-commit judgement
 * must not do. What is here is read off the stream's terminal parts instead.
 *
 * There is no content, and that is a fact rather than an omission — any content
 * part would have committed the attempt and put it beyond retry. Result
 * conditions here are therefore effectively finish-reason-shaped, a ceiling
 * inherent to streaming rather than to this implementation, and this type says
 * so by declaring no content at all.
 */
export type StreamTextCommitResult = {
  finishReason: CallFinishReason;
  usage: CallLanguageModelUsage;
  providerMetadata: ProviderMetadata | undefined;
};
