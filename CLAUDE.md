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
- **`src/model/<family>-model/`**: the model-level API, one folder per family, each with its own `conditions/`.
- **`src/call/<entry-point>/`**: the call-level API, one folder per entry point (`generate-text`, `stream-text`, `embed`, `embed-many`, `generate-image`), each holding its function, its `types.ts` (its commit result), its `conditions/`, and its own `.test.ts` / `.test-d.ts`. Shared beside them: one loop (`run-retry-loop.ts`), the machinery (`retryable-calls.ts` — two deadline strategies plus `defineRetryableCall`), and `conditions/result.ts`. Behavior belonging to the shared loop rather than to one entry point is tested once in `run-retry-loop.test.ts`. Both layers share the internals under `src/internal/`.
- **`src/call/types.ts`**: the layer's types, carrying what a condition judges only as `COMMIT`. Which result that is belongs to the entry point, so each declares its own in its folder's `types.ts` — `GenerateTextCommitResult`, `StreamTextCommitResult`, and so on. `CallRetryContext` is deliberately a **different type** from the model layer's `ModelRetryContext` — that is the only thing stopping a condition written for one layer from typechecking against the other.
- **`Condition<MODEL, LAYER, COMMIT>`** in `src/internal/conditions/`: `LAYER` defaults to `'model'`, so the model-layer conditions are unchanged and both layers share one implementation of the error API and the combinators. `COMMIT` defaults to `unknown`, and that is the **permissive** end — it surfaces only through the predicate, making `Condition` contravariant in it, so a condition that never reads the result fits every entry point while a `result()` condition fits only its own. Getting this backwards (`never`) makes error conditions assignable nowhere.

### Adding an entry point

A new folder under `src/call/`, plus a hand-written `exports` entry in package.json. Nothing central changes — deliberately. If a change would make a new entry point edit a shared conditional type or registry, it is the wrong change; `PLAN.md` §15.3 records that design being rejected for exactly that reason.

`defineRetryableCall<MODEL, ARGS, RESULT, COMMIT = RESULT>`: the default says "what the caller receives is what conditions judge", which holds for four of the five. `streamText` overrides it because its result is a set of promises that settle only on consumption, and consuming it is what a pre-commit judgement must not do. Do not assume that is the only such case.

### The export map is hand-written

`src/` splits at the top into the two retry layers and **neither segment appears in the published path** (`src/call/generate-text/` → `ai-retry/generate-text`, `src/model/language-model/` → `ai-retry/language-model`). So the map cannot be derived from the layout, `tsdown`'s `exports: true` is off, and package.json carries it. `publint` and `attw` check that listed entries resolve; `src/index.test.ts` checks the other direction, that every built `index.ts` is listed.

`PLAN.md` is the design record for the call-level API: what was measured, what was rejected, and why. Read it before changing the signatures, the `INPUT`/`OVERRIDE`/`COMMIT` generics, or the export shape — several of the obvious simplifications were tried and are documented as failures. §14 describes a family-keyed design with a discriminated result union that §15 supersedes; where they disagree, §15 is what shipped.

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
- The call-level functions cover `generateText`, `streamText`, `embed`, `embedMany`, `generateImage`, each published at `ai-retry/<function>` with its conditions at `ai-retry/<function>/conditions`; the object entry points are deliberately out of scope (`streamObject`'s `fullStream` is not a fresh tee, so reading it for commit detection destroys the caller's stream)
- Streaming retry support with limitations: retries only possible before content starts flowing

## Key Implementation Details

- **Retry Loop Prevention**: Uses model keys (`provider/modelId`) to track attempts per model
- **Two Retry Types**: Error-based (API failures) and result-based (content filtering, schema mismatches). Result-based retries work for every entry point at the call level; at the model level they are language-only, since the embedding and image wrappers have no result branch.
- **Results reach conditions by identity**: the call layer passes the SDK's own object through, never a copy. This matters because SDK results expose most of themselves through prototype getters, so anything rebuilt with a spread silently yields `undefined` for `text` and `toolCalls`. An earlier design tagged results with an `operation` discriminant via a `Proxy` to avoid exactly that; per-entry-point conditions removed the need for the tag, and the `Proxy` with it.
- **State Management**: `RetryableModel` class maintains current model and tracks all attempts
- **Error Handling**: Throws `RetryError` when all retries fail, original error when no retries attempted

## Type naming

Types belonging to one retry layer carry its prefix: `ModelRetryContext` / `CallRetryContext`, `ModelRetryAttempt` / `CallRetryAttempt`, `ModelRetryable` / `CallRetryable`, `ModelCallOptions` (provider options) / `CallArgs` (entry point args), and so on. `Retry`, `OnRetryOverrides`, `Reset` and `RetryTelemetrySettings` are shared and carry no prefix.

The call layer's results are the exception, because they belong to an entry point rather than to the layer: `GenerateTextCommitResult`, `StreamTextCommitResult`, `EmbedCommitResult`, `EmbedManyCommitResult`, `GenerateImageCommitResult`, each in its own folder's `types.ts`. `*CommitResult` reads as "what a result condition judges at the moment the attempt would commit"; for four of the five it is an alias of the SDK's own result type, and the name is the contract rather than the shape.

The unprefixed names (`RetryContext`, `Retryable`, `CallOptions`, …) predate the call layer and survive as deprecated aliases in `src/types.ts`. Use the prefixed ones in new code. `src/types.test-d.ts` pins each alias to be the _same_ type as its replacement, so one cannot drift from the other while both exist.

## Type tests

Prefer `toEqualTypeOf` over `toMatchTypeOf`. The latter is deprecated (expect-type >= 1.2, use `toExtend`) and, more importantly, only asserts _assignability_ — it passes against `any`, so it silently stops catching anything the moment a signature degrades. A mutation making `Condition.switch` return `any` was caught by 6 assertions under `toMatchTypeOf` and by 24 under `toEqualTypeOf`.

Note `ModelRetryable<M>` and `CallRetryable<M>` have different `INPUT` defaults (the provider-level overrides vs `never`). `.switch()`/`.retry()` leave `INPUT` unbound, so their exact return is the `never` instantiation — assert `ModelRetryable<M, never>`, not `ModelRetryable<M>`.
