# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Development Commands

- **Build**: `pnpm build` - Uses tsdown to build the TypeScript project
- **Test**: `pnpm test` - Runs tests with Vitest (120s timeout configured)
- **Lint**: `pnpm lint` - Uses Biome for linting and formatting with auto-fix
- **Type Check**: Use TypeScript compiler directly (`npx tsc --noEmit`) for type checking
- **Single Test**: `pnpm test <test-pattern>` - Run specific test files or patterns

## Architecture Overview

This is an AI SDK retry library that provides intelligent fallback mechanisms for AI model failures. The core architecture consists of:

### Core Components

- **`create-retryable-model.ts`**: Main factory function that creates a retryable model wrapper implementing `LanguageModelV2`
- **`RetryableModel` class**: Wraps any AI model and handles retry logic with state tracking across attempts
- **`src/retryables/`**: Individual retry handlers for specific error conditions

### Retry System Design

The retry system uses a functional approach where:
1. Each retryable handler is a function that receives a `RetryContext` and returns a `RetryModel` or `undefined`
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
- Supports `generateText`, `generateObject`, `streamText`, and `streamObject`
- Streaming retry support with limitations: retries only possible before content starts flowing

## Key Implementation Details

- **Retry Loop Prevention**: Uses model keys (`provider/modelId`) to track attempts per model
- **Two Retry Types**: Error-based (API failures) and result-based (content filtering, schema mismatches)
- **State Management**: `RetryableModel` class maintains current model and tracks all attempts
- **Error Handling**: Throws `RetryError` when all retries fail, original error when no retries attempted