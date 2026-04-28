/**
 * Experimental: composable `condition().action()` retryable API.
 *
 * Status: under evaluation. Not exported from the package root. Opt in with
 * `import { httpStatus, finishReason, ... } from 'ai-retry/retryables/experimental'`.
 *
 * Mapping from the stable retryables to this shape:
 *
 *   contentFilterTriggered(fallback)
 *     → or(
 *         error(e => APICallError.isInstance(e)
 *           && (e.data as any)?.error?.code === 'content_filter'),
 *         finishReason('content-filter'),
 *       ).switch({ model: fallback })
 *
 *   requestTimeout(fallback)
 *     → timeout().switch({ model: fallback, timeout: 60_000 })
 *
 *   requestNotRetryable(fallback)
 *     → error.isRetryable(false).switch({ model: fallback })
 *
 *   schemaMismatch(fallback)
 *     → schemaInvalid().switch({ model: fallback })
 *
 *   serviceOverloaded(fallback)
 *     → httpStatus(529, 'overloaded').switch({ model: fallback })
 *
 *   serviceUnavailable(fallback)
 *     → error.statusCode(503).switch({ model: fallback })
 *
 *   noImageGenerated(fallback)
 *     → noImage().switch({ model: fallback })
 *
 *   retryAfterDelay({ delay, backoffFactor })
 *     → error.isRetryable(true).retry({ delay, backoffFactor })
 *     // .retry() always honors Retry-After / Retry-After-Ms headers when
 *     // the error carries them.
 *
 * Note on `error.isRetryable(true)`:
 *   The AI SDK's `APICallError` defaults `isRetryable=true` for status codes
 *   408, 409, 429, and any 5xx. Network errors (connection refused, DNS,
 *   fetch failures) are explicitly marked retryable too. Some providers
 *   override the default: Anthropic flips it to true on
 *   `error.type === 'overloaded_error'`; OpenAI / HuggingFace pin 400s to
 *   false. So `error.isRetryable(true)` is broader than a status-code list:
 *   it picks up network errors and provider overrides automatically.
 */

export * from './aborted.js';
export * from './and.js';
export * from './condition.js';
export * from './error.js';
export * from './finish-reason.js';
export * from './http-status.js';
export * from './no-image.js';
export * from './not.js';
export * from './or.js';
export * from './result.js';
export * from './schema-invalid.js';
export * from './timeout.js';
