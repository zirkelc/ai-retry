# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Development Commands

- **Build**: `pnpm build` - Uses tsdown to build the TypeScript project
- **Test**: `pnpm test` - Runs tests with Vitest (120s timeout configured)
- **Lint**: `pnpm lint` - Uses Biome for linting and formatting with auto-fix
- **Type Check**: `pnpm types` - Validates package types using @arethetypeswrong/cli

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

- Built for AI SDK v2 (`@ai-sdk/provider`, `@ai-sdk/provider-utils`)
- Uses Biome for code formatting (single quotes, semicolons, trailing commas)
- TypeScript with strict configuration
- Vitest for testing with MSW for HTTP mocking