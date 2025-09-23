### ai-retry

Intelligent retry and fallback mechanisms for AI SDK models. Automatically handle API failures, content filtering, timeouts, and schema mismatches by switching between different AI models.

#### How?

`ai-retry` wraps the provided base model with a set of retry conditions (retryables). When a request fails due to specific errors OR when a successful response has certain characteristics (like content filtering), it iterates through the given retryables to find a suitable fallback model. It automatically tracks which models have been tried and how many attempts have been made to prevent infinite loops.

The system supports two types of retries:
- **Error-based retries**: Triggered when the model throws an error (timeouts, API errors, etc.)
- **Result-based retries**: Triggered when the model returns a successful response that needs retrying (e.g., content filtering, schema mismatches)

### Installation

This library only supports AI SDK v5.

> [!WARNING]  
> `ai-retry` is in alpha stage and the API may change in future releases.

```bash
npm install ai-retry@alpha
```

### Usage

Create a retryable model by providing a base model and a list of retryables or fallback models.

> [!WARNING]  
> `ai-retry` currently only supports `generateText` and `generateObject` calls.
> Streaming via `streamText` and `streamObject` is not supported yet.

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

#### Retry Errors and Results

`ai-retry` handles two distinct retry scenarios:

##### Error-based Retries

Triggered when the model throws an error during `generateText` or `generateObject` calls:

```typescript
import { requestTimeout, requestNotRetryable } from 'ai-retry/retryables';

const retryableModel = createRetryable({
  model: azure('gpt-4'),
  retries: [
    requestTimeout(azure('gpt-4-mini')), // Timeout error
    requestNotRetryable(openai('gpt-4')), // Non-retryable API error
  ],
});
```

##### Result-based Retries  

Triggered when the model returns a successful response that needs retrying:

```typescript
import { contentFilterTriggered } from 'ai-retry/retryables';

const retryableModel = createRetryable({
  model: azure('gpt-4-mini'),
  retries: [
    contentFilterTriggered(openai('gpt-4-mini')), // finishReason: 'content-filter'
  ],
});
```

#### Retryables

A retryable is a function that receives the current attempt and determines whether to retry with a different model based on the error/result and any previous attempts. 
There are several built-in retryables:

- `contentFilterTriggered`: Content filter was triggered based on the prompt or completion.
<!-- - `responseSchemaMismatch`: Structured output validation failed. -->
- `requestTimeout`: Request timeout occurred.
- `requestNotRetryable`: Request failed with a non-retryable error.

By default, each retryable will only attempt to retry once per model to avoid infinite loops. You can customize this behavior by returning a `maxAttempts` value from your retryable function.

##### Content Filter Triggered

Automatically switch to a different model when content filtering blocks your request.

```typescript
import { contentFilterTriggered } from 'ai-retry/retryables';

const retryableModel = createRetryable({
  model: azure('gpt-4-mini'),
  retries: [
    contentFilterTriggered(openai('gpt-4-mini')), // Try OpenAI if Azure filters
  ],
});
```

<!--
##### Response Schema Mismatch

Retry with different models when structured output validation fails:

```typescript
import { responseSchemaMismatch } from 'ai-retry/retryables';

const retryableModel = createRetryable({
  model: azure('gpt-4-mini'),
  retries: [
    responseSchemaMismatch(azure('gpt-4')), // Try full model for better structured output
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
-->

##### Request Timeout

Handle timeouts by switching to potentially faster models.

> [!NOTE] 
> You need to use an `abortSignal` with a timeout on your request. 

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

##### Request Not Retryable

Handle cases where the base model fails with a non-retryable error.

> [!NOTE] 
> You can check if an error is retryable with the `isRetryable` property on an [`APICallError`](https://ai-sdk.dev/docs/reference/ai-sdk-errors/ai-api-call-error#ai_apicallerror).


```typescript
import { requestNotRetryable } from 'ai-retry/retryables';

const retryable = createRetryable({
  model: azure('gpt-4-mini'),
  retries: [
    requestNotRetryable(openai('gpt-4')), // Switch providers entirely
  ],
});
```

##### Custom Retryables

Create your own retryables for specific use cases:

```typescript
import type { Retryable } from 'ai-retry';

const customRetry: Retryable = (context) => {
  const { current, attempts, totalAttempts } = context;
  
  // Your custom logic here
  if (shouldRetryWithDifferentModel(current.error)) {
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

#### Default Fallback

If you always want to fallback to a different model on any error, you can simply provide a list of models.

```typescript
const retryableModel = createRetryable({
  model: azure('gpt-4'),
  retries: [
    openai('gpt-4'), 
    anthropic('claude-3-haiku-20240307')
  ],
});
```

#### All Retries Failed

If all retry attempts failed, a `RetryError` is thrown containing all individual errors.
If no retry was attempted (e.g. because all retryables returned `undefined`), the original error is thrown directly.

```typescript
import { RetryError } from 'ai';

const retryableModel = createRetryable({
  model: azure('gpt-4'),
  retries: [
    openai('gpt-4'), 
    anthropic('claude-3-haiku-20240307')
  ],
});

try {
  const result = await generateText({
    model: retryableModel,
    prompt: 'Hello world!',
  });
} catch (error) {
  // RetryError is an official AI SDK error
  if (error instanceof RetryError) {
    console.error('All retry attempts failed:', error.errors);
  } else {
    console.error('Request failed:', error);
  }
}
```

#### Logging

You can use the following callbacks to log retry attempts and errors:
- `onError` is invoked if an error occurs. 
- `onRetry` is invoked before attempting a retry.

```typescript
const retryableModel = createRetryable({
  model: openai('gpt-4-mini'),
  retries: [/* your retryables */],
  onError: (context) => {
    console.log(`Attempt ${context.totalAttempts} with ${context.current.model.provider}/${context.current.model.modelId} failed:`, context.current.error);
  },
  onRetry: (context) => {
    console.log(`Retrying with model ${context.current.model.provider}/${context.current.model.modelId}...`);
  },
});
```

### API Reference

#### `createRetryable(options: CreateRetryableOptions): LanguageModelV2`

Creates a retryable language model.

```ts
interface CreateRetryableOptions {
  model: LanguageModelV2;
  retries: Array<Retryable | LanguageModelV2>;
  onError?: (context: RetryContext) => void;
  onRetry?: (context: RetryContext) => void; 
}
```

#### `Retryable`

A `Retryable` is a function that receives a `RetryContext` with the current error or result and model and all previous attempts.
It should evaluate the error/result and decide whether to retry by returning a `RetryModel` or to skip by returning `undefined`.

```ts
type Retryable = (context: RetryContext) => RetryModel | Promise<RetryModel> | undefined;
```

#### `RetryModel`

A `RetryModel` specifies the model to retry and an optional `maxAttempts` to limit how many times this model can be retried.
By default, each retryable will only attempt to retry once per model. This can be customized by setting the `maxAttempts` property.

```typescript
interface RetryModel {
  model: LanguageModelV2;
  maxAttempts?: number;
}
```

#### `RetryContext`

The `RetryContext` object contains information about the current attempt and all previous attempts.

```typescript
interface RetryContext {
  current: RetryAttempt;
  attempts: Array<RetryAttempt>;
  totalAttempts: number;
}
```

#### `RetryAttempt`

A `RetryAttempt` represents a single attempt with a specific model, which can be either an error or a successful result that triggered a retry.

```typescript
type RetryAttempt = 
  | { type: 'error'; error: unknown; model: LanguageModelV2 }
  | { type: 'result'; result: LanguageModelV2Generate; model: LanguageModelV2 };

// Type guards for discriminating attempts
function isErrorAttempt(attempt: RetryAttempt): attempt is RetryErrorAttempt;
function isResultAttempt(attempt: RetryAttempt): attempt is RetryResultAttempt;
```

### License

MIT
