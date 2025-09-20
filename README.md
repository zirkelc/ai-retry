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
  // Base model
  model: azure('gpt-4-mini'), 
  // Retry strategies
  retries: [
    contentFilterTriggered(openai('gpt-4-mini')), 
    requestTimeout(azure('gpt-4')), 
    openai('gpt-4-mini'),
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

Handle cases where the base model fails with a non-retryable error (e.g., unsupported features):

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

Create your own retryables for specific use cases:

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
  retries: [/* your retryables */],
  onError: (context) => {
    console.log(`Attempt ${context.totalAttempts} failed:`, context.error);
    console.log(`Tried models:`, Array.from(context.triedModels.keys()));
  },
});
```

## API Reference

### `createRetryable(options: CreateRetryableOptions): LanguageModelV2`

Creates a retryable language model.

```ts
interface CreateRetryableOptions {
  model: LanguageModelV2;
  retries: Array<Retryable | LanguageModelV2>;
  onError?: (context: RetryContext) => void;
}
```

### `Retryable`

A `Retryable` is a function that receives a `RetryContext` with the current error and model and all previously tried models.
It should evaluate the error and decide whether to retry by returning a new model or to skip by returning `undefined`.

```ts
type Retryable = (context: RetryContext) => RetryModel | Promise<RetryModel> | undefined;
```

### `RetryModel`

A `RetryModel` specifies the model to retry with and an optional `maxAttempts` to limit how many times this model can be retried.

```typescript
interface RetryModel {
  model: LanguageModelV2;
  maxAttempts?: number;
}
```

### `RetryContext`

The `RetryContext` object contains information about the current error and previously tried models.

```typescript
interface RetryContext {
  error: unknown;                       
  baseModel: LanguageModelV2;           
  currentModel: LanguageModelV2;        
  triedModels: Map<string, RetryState>; 
  totalAttempts: number;                
}
```

### `RetryState`

The `RetryState` tracks the state of each model that has been tried, including the number of attempts and any errors encountered. The `modelKey` is a unique identifier for the model instance to keep track of models without relying on object reference equality.

```typescript
interface RetryState {
  modelKey: string;
  model: LanguageModelV2;
  attempts: number;
  errors: Array<unknown>;
}
```

## Requirements

- AI SDK v2.0+
- Node.js 16+

## License

MIT
