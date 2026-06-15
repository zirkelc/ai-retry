# Migrating from v1 to v2

v2 promotes the **condition-based API** (previously shipped under `ai-retry/experimental/*`) to the package's primary, stable API. This guide explains how to move from the v1 function-style retryables to the new condition API.

> [!TIP]
> Nothing forces an immediate rewrite. The v1 function-style retryables and the root `createRetryable` still work in v2 (they are deprecated, not removed). You can migrate incrementally.

## Version compatibility

| Version | AI SDK | Primary API |
| --- | --- | --- |
| `ai-retry@0.x` | AI SDK v5 | function-style retryables |
| `ai-retry@1.x` | AI SDK v6 | function-style retryables |
| `ai-retry@2.x` | AI SDK v6 | condition-based retryables |

There is **no AI SDK version bump** between v1 and v2: both target AI SDK v6 and the same peer-dependency ranges. The only change is the public API surface.

If you are not ready to migrate, pin to `ai-retry@1` and read the [v1 README](https://github.com/zirkelc/ai-retry/blob/v1/README.md).

## What still works

If you wrote code like this in v1, it **still compiles and runs unchanged in v2**:

```ts
import { createRetryable } from 'ai-retry';
import { contentFilterTriggered, serviceOverloaded } from 'ai-retry/retryables';

const model = createRetryable({
  model: openai('gpt-4o'),
  retries: [
    contentFilterTriggered(anthropic('claude-sonnet-4-5')),
    serviceOverloaded(anthropic('claude-sonnet-4-5')),
  ],
});
```

The root `createRetryable` and every function from `ai-retry/retryables` are kept for backwards compatibility, but they are **`@deprecated`**. Your editor/linter will flag them. The two steps below convert this to the new condition API.

## Step 1: Switch the factory

```diff
- import { createRetryable } from 'ai-retry';
+ import { createRetryableModel } from 'ai-retry/language-model';

- createRetryable({ model, retries: [...] })
+ createRetryableModel({ model, retries: [...] })
```

Use the entry point that matches your model: `ai-retry/language-model`, `ai-retry/embedding-model`, or `ai-retry/image-model`. Unlike the auto-detecting root `createRetryable` (which only resolves bare gateway strings as language models), each per-family factory is typed for its family and resolves gateway strings for that family.

## Step 2: Replace each retryable with a condition

Every deprecated retryable maps to a condition built from the matchers exported by `ai-retry/<family>-model` (also available on the `ai-retry/<family>-model/conditions` subpath). `.switch({ model })` switches to a fallback for a single attempt (matching the old default); `.retry({ ... })` retries the same model.

| Deprecated retryable | Matches | Condition equivalent |
| --- | --- | --- |
| `contentFilterTriggered(fallback)` | result `finishReason === 'content-filter'` | `finishReason('content-filter').switch({ model: fallback })` |
| `requestTimeout(fallback)` | `TimeoutError` (from `AbortSignal.timeout()`) | `timeout().switch({ model: fallback, timeout: 60_000 })` |
| `requestNotRetryable(fallback)` | `APICallError` with `isRetryable === false` | `error.isRetryable(false).switch({ model: fallback })` |
| `retryAfterDelay({ delay, backoffFactor, maxAttempts })` | `APICallError` with `isRetryable === true`; honors `Retry-After` headers | `error.isRetryable(true).retry({ delay, backoffFactor, maxAttempts })` |
| `schemaMismatch(fallback)` | result text fails JSON-schema validation | `schemaInvalid().switch({ model: fallback })` |
| `serviceOverloaded(fallback)` | status `529` | `httpStatus(529).switch({ model: fallback })` |
| `serviceUnavailable(fallback)` | status `503` | `httpStatus(503).switch({ model: fallback })` |
| `noImageGenerated(fallback)` | `NoImageGeneratedError` | `noImage().switch({ model: fallback })` (from `ai-retry/image-model`) |

Full example:

```diff
- import { createRetryable } from 'ai-retry';
- import { serviceOverloaded, requestNotRetryable } from 'ai-retry/retryables';
+ import { createRetryableModel, httpStatus, error } from 'ai-retry/language-model';

  const fallback = anthropic('claude-sonnet-4-5');

- const model = createRetryable({
+ const model = createRetryableModel({
    model: openai('gpt-4o'),
    retries: [
-     serviceOverloaded(fallback),
-     requestNotRetryable(fallback),
+     httpStatus(529, 'overloaded').switch({ model: fallback }),
+     error.isRetryable(false).switch({ model: fallback }),
    ],
  });
```

## Behavior parity notes

- `.switch({ model })` switches to the fallback for a single attempt — the same as the old retryables' implicit `maxAttempts: 1`.
- `.retry()` defaults to `maxAttempts: 2` (one original + one retry) and honors `Retry-After` / `Retry-After-Ms` headers (capped at 60s), matching `retryAfterDelay`. `maxAttempts: 1` is rejected — use `.switch()` for a single different-model attempt.
- The condition matchers are the exact same implementations that powered the deprecated retryables, so matching semantics are unchanged.

## Reference

- [v2 README](./README.md) — full condition API documentation
- [v1 README](https://github.com/zirkelc/ai-retry/blob/v1/README.md) — function-style retryable documentation
