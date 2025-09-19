# ai-retry

🚧 **WORK IN PROGRESS - DO NOT USE IN PRODUCTION**

Intelligent retry and fallback mechanisms for AI SDK models. Automatically handle API failures, content filtering, timeouts, and schema mismatches by switching between different AI models.

## How It Works

`ai-retry` wraps the provided base model with a set of retry conditions (retryables). When a request fails due to specific errors (like content filtering or timeouts), it iterates through the given retryables to find a suitable fallback model. It tracks which models have been tried and how many attempts have been made to prevent infinite loops.

## Installation

```bash
npm install ai-retry
# or
pnpm add ai-retry
# or
yarn add ai-retry
```

## Usage

```typescript
import { azure } from '@ai-sdk/azure';
import { openai } from '@ai-sdk/openai';
import { generateText } from 'ai';
import { createRetryable } from 'ai-retry';
import { contentFilterTriggered, requestTimeout } from 'ai-retry/retryables';

// Create a retryable model
const retryableModel = createRetryable({
  model: azure('gpt-4-mini'), // Base model
  retries: [
    contentFilterTriggered(openai('gpt-4-mini')), // Switch if content filtered
    requestTimeout(azure('gpt-4')), // Switch on timeout
    openai('gpt-4-mini'), // Final fallback to OpenAI
  ],
});

// Use like any other AI SDK model
const result = await generateText({
  model: retryableModel,
  prompt: 'Hello world!',
  abortSignal: AbortSignal.timeout(10_000), 
});
```

### Default Fallback

If you always want to fallback to a different model on any error, you can simply provide a list of models:

```typescript
const retryableModel = createRetryable({
  model: azure('gpt-4'),
  retries: [
    openai('gpt-4'), 
    anthropic('claude-3-haiku-20240307')
  ],
});
```

## Retryables

A retryable is a function that receives an error and determines whether to retry with a different model based on the error and context of previous attempts. 
`ai-retry` includes several built-in retryables:

- `contentFilterTriggered`: Automatically switch to a different model when content filtering blocks your request.
- `responseSchemaMismatch`: Retry with different models when structured output validation fails.
- `requestTimeout`: Handle timeouts by switching to potentially faster models.
- `requestNotRetryable`: Handle cases where the primary model cannot process the request.

By default, each retryable will only attempt to retry once per model to avoid infinite loops. You can customize this behavior by returning a `maxAttempts` value from your retryable function.

### Content Filter Triggered

Automatically switch to a different model when content filtering blocks your request:

```typescript
import { contentFilterTriggered } from 'ai-retry/retryables';

const retryableModel = createRetryable({
  model: azure('gpt-4-mini'),
  retries: [
    contentFilterTriggered(openai('gpt-4-mini')), // Try OpenAI if Azure filters
  ],
});
```

### Response Schema Mismatch

Retry with different models when structured output validation fails:

```typescript
import { responseSchemaMismatch } from 'ai-retry/retryables';

const retryableModel = createRetryable({
  model: azure('gpt-4-mini'),
  retries: [
    responseSchemaMismatch(azure('gpt-4')), // Try full model for better structure output
  ],
});

const result = await generateObject({
  model: retryableModel,
  schema: z.object({
    recipe: z.object({
      name: z.string(),
      ingredients: z.array(z.object({ name: z.string(), amount: z.string() })),
      steps: z.array(z.string()),
    }),
  }),
  prompt: 'Generate a lasagna recipe.',
});
```

### Request Timeout

Handle timeouts by switching to potentially faster models:

```typescript
import { requestTimeout } from 'ai-retry/retryables';

const retryableModel = createRetryable({
  model: azure('gpt-4'),
  retries: [
    requestTimeout(azure('gpt-4-mini')), // Use faster model on timeout
  ],
});

const result = await generateText({
  model: retryableModel,
  prompt: 'Write a vegetarian lasagna recipe for 4 people.',
  abortSignal: AbortSignal.timeout(10_000), 
});
```

### Request Not Retryable

Handle cases where the primary model cannot process the request:

```typescript
import { requestNotRetryable } from 'ai-retry/retryables';

const retryable = createRetryable({
  model: azure('gpt-4-mini'),
  retries: [
    requestNotRetryable(openai('gpt-4')), // Switch providers entirely
  ],
});
```

### Custom Retryables

Create your own retry handlers for specific use cases:

```typescript
import type { Retryable } from 'ai-retry';

const customRetry: Retryable = (context) => {
  const { error, triedModels, totalAttempts } = context;
  
  // Your custom logic here
  if (shouldRetryWithDifferentModel(error)) {
    return {
      model: myFallbackModel,
      maxAttempts: 3,
    };
  }
  
  return undefined; // Don't retry
};

const retryable = createRetryable({
  model: azure('gpt-4-mini'),
  retries: [customRetry],
});
```

### Error Monitoring

Track retry attempts and errors:

```typescript
const retryable = createRetryable({
  model: primaryModel,
  retries: [/* your retries */],
  onError: (context) => {
    console.log(`Attempt ${context.totalAttempts} failed:`, context.error);
    console.log(`Tried models:`, Array.from(context.triedModels.keys()));
  },
});
```

## API Reference

### `createRetryable(options)`

Creates a retryable model wrapper.

**Options:**
- `model`: Primary AI model to use
- `retries`: Array of retry strategies or fallback models
- `onError`: Optional callback for error monitoring

**Returns:** A `LanguageModelV2` compatible model

### Retry Context

Each retry handler receives a context object:

```typescript
interface RetryContext {
  error: unknown;              // The error that triggered the retry
  baseModel: LanguageModelV2;  // Original model
  currentModel: LanguageModelV2; // Model that just failed
  triedModels: Map<string, RetryState>; // Previously tried models
  totalAttempts: number;       // Total attempts across all models
}
```

## Requirements

- AI SDK v2.0+
- Node.js 16+

## License

MIT
