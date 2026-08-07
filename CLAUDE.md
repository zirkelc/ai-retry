# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Development Commands

- **Build**: `pnpm build` - Uses tsdown to build the TypeScript project
- **Test**: `pnpm test` - Runs tests with Vitest (120s timeout configured)
- **Coverage**: `pnpm test:coverage` - Vitest + v8 coverage, gated by thresholds. The provider version must match vitest **exactly** (`@vitest/coverage-v8@4.0.18`); a `^4` range resolves higher and crashes with `does not provide an export named 'BaseCoverageProvider'`. Type-only modules and re-export-only barrels are excluded — they emit no runtime and would report 0% however well the types are exercised.
- **Lint**: `pnpm lint` - Uses Biome for linting and formatting with auto-fix
- **Type Check**: Use TypeScript compiler directly (`npx tsc --noEmit`) for type checking
- **Single Test**: `pnpm test <test-pattern>` - Run specific test files or patterns

## Architecture Overview

This is an AI SDK retry library that provides intelligent fallback mechanisms for AI model failures. The core architecture consists of:

Two layers offer retries, and they differ in _where_ the retry sits relative to the call:

- **Model-level** (`createRetryableModel`) retries inside `doGenerate`/`doStream`, **below** the entry point. Structurally blind to anything living on the call itself — a `timeout` argument or an inbound `abortSignal` — because by the time one fires the SDK has torn the call down.
- **Call-level** (`src/call/`) re-runs the **whole** entry point with the next model, which is the only way those recover.

### Core Components

- **`create-retryable-model.ts`**: Main factory function that creates a retryable model wrapper implementing `LanguageModelV2`
- **`RetryableModel` class**: Wraps any AI model and handles retry logic with state tracking across attempts
- **`src/retryables/`**: Individual retry handlers for specific error conditions
- **`src/call/`**: The call-level API. One loop (`run-retry-loop.ts`), the shared machinery (`retryable-calls.ts` — two deadline strategies plus `defineRetryableCall`), and one module per entry point under `call/<family>-model/functions/` holding its row, its hand-written signature and its export. Each entry point owns its `.test.ts` and `.test-d.ts` beside it; behavior that belongs to the shared loop rather than to one entry point is tested once in `run-retry-loop.test.ts`. Both layers share the internals under `src/internal/`.
- **`src/call/types.ts` + `src/call/guards.ts`**: what a call-level condition judges. The result is the entry point's own, tagged with its `operation`, so each family is a discriminated union (`generateText` | `streamText`, `embed` | `embedMany`, `generateImage`) narrowed by the `is*Result` guards. `CallRetryContext` is deliberately a **different type** from the model layer's `RetryContext` — that is the only thing stopping a condition written for one layer from typechecking against the other. The split mirrors the root: `types.ts` type-only, `guards.ts` the narrowing runtime.
- **`src/call/conditions/` + `src/call/<family>-model/conditions/`**: the call-layer condition API, published as `ai-retry/call/<family>-model/conditions`. `Condition<MODEL, LAYER>` in `src/internal/conditions/` carries a layer tag defaulting to `'model'`, so the model-layer conditions are unchanged and both layers share one implementation of the error API and the combinators.

`PLAN.md` is the design record for the call-level API: what was measured, what was rejected, and why. Read it before changing the signatures, the `INPUT`/`OVERRIDE` generics, or the result union — several of the obvious simplifications were tried and are documented as failures, and §14.6 records two probe findings that the implementation later contradicted.

### Retry System Design

The retry system uses a functional approach where:

1. Each retryable handler is a function that receives a retry context (`ModelRetryContext` below a model, `CallRetryContext` around a call) and returns a `Retry` or `undefined`
2. The context includes error details, tried models map, and attempt counts
3. Retry handlers can specify different fallback models and max attempts per model
4. The system prevents infinite loops by tracking which models have been tried

### Built-in Retryable Handlers

Located in `src/retryables/`:

- **content-filter-triggered**: Switches models when content filtering blocks responses
- **request-timeout**: Handles timeout errors
- **request-not-retryable**: Handles non-retryable request errors
- **response-schema-mismatch**: Switches models for schema validation failures
- **service-overloaded**: Handles HTTP 529 service overloaded errors
- **anthropic-service-overloaded**: Anthropic-specific overload handling for both HTTP 529 and 200 OK responses

### Usage Pattern

```typescript
const retryableModel = createRetryable({
  model: primaryModel,
  retries: [
    contentFilterTriggered(fallbackModel),
    requestTimeout(alternateModel),
    // ... other handlers
  ],
});
```

## Dependencies

- Built for AI SDK v5 (`@ai-sdk/provider`, `@ai-sdk/provider-utils`)
- Uses Biome for code formatting (single quotes, semicolons, trailing commas)
- TypeScript with strict configuration using @total-typescript/tsconfig
- Vitest for testing with MSW for HTTP mocking
- Model-level wrappers support `generateText`, `generateObject`, `streamText`, and `streamObject`
- The call-level functions cover `generateText`, `streamText`, `embed`, `embedMany`, `generateImage`; the object entry points are deliberately out of scope (`streamObject`'s `fullStream` is not a fresh tee, so reading it for commit detection destroys the caller's stream)
- Streaming retry support with limitations: retries only possible before content starts flowing

## Key Implementation Details

- **Retry Loop Prevention**: Uses model keys (`provider/modelId`) to track attempts per model
- **Two Retry Types**: Error-based (API failures) and result-based (content filtering, schema mismatches). Result-based retries work for every family at the call level; at the model level they are language-only, since the embedding and image wrappers have no result branch.
- **Tagging results**: `tagResult` in `src/call/tag-result.ts` is a `Proxy`, not a copy. The SDK's results expose most of themselves through prototype getters, so `{ operation, ...result }` silently yields `undefined` for `text` and `toolCalls`.
- **State Management**: `RetryableModel` class maintains current model and tracks all attempts
- **Error Handling**: Throws `RetryError` when all retries fail, original error when no retries attempted

## Type naming

Types belonging to one retry layer carry its prefix: `ModelRetryContext` / `CallRetryContext`, `ModelRetryAttempt` / `CallRetryAttempt`, `ModelRetryable` / `CallRetryable`, `ModelCallOptions` (provider options) / `CallArgs` (entry point args), and so on. `Retry`, `OnRetryOverrides`, `Reset` and `RetryTelemetrySettings` are shared and carry no prefix.

The unprefixed names (`RetryContext`, `Retryable`, `CallOptions`, …) predate the call layer and survive as deprecated aliases in `src/types.ts`. Use the prefixed ones in new code. `src/types.test-d.ts` pins each alias to be the _same_ type as its replacement, so one cannot drift from the other while both exist.

## Type tests

Prefer `toEqualTypeOf` over `toMatchTypeOf`. The latter is deprecated (expect-type >= 1.2, use `toExtend`) and, more importantly, only asserts _assignability_ — it passes against `any`, so it silently stops catching anything the moment a signature degrades. A mutation making `Condition.switch` return `any` was caught by 6 assertions under `toMatchTypeOf` and by 24 under `toEqualTypeOf`.

Note `ModelRetryable<M>` and `CallRetryable<M>` have different `INPUT` defaults (the provider-level overrides vs `never`). `.switch()`/`.retry()` leave `INPUT` unbound, so their exact return is the `never` instantiation — assert `ModelRetryable<M, never>`, not `ModelRetryable<M>`.
