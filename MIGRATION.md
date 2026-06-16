# Migrating to the condition API

The **condition-based API** (per-model entry points + `createRetryableModel`) is the recommended way to configure retries. This guide shows how to move existing code from the function-style retryables to the condition API.

> [!TIP]
> Nothing forces an immediate rewrite. The function-style retryables and the root `createRetryable` are deprecated but still work — they ship in the same package as the condition API, so you can migrate incrementally.

## What still works

The following keeps compiling and running unchanged:

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

- [README](./README.md) — full condition API documentation
- [Earlier README](https://github.com/zirkelc/ai-retry/blob/v1/README.md) — full function-style retryable documentation
