<a href="https://www.npmjs.com/package/ai-retry" alt="ai-retry"><img src="https://img.shields.io/npm/dt/ai-retry?label=ai-retry"></a> <a href="https://github.com/zirkelc/ai-retry/actions/workflows/ci.yml" alt="CI"><img src="https://img.shields.io/github/actions/workflow/status/zirkelc/ai-retry/ci.yml?branch=main"></a>

# ai-retry: Retry and fallback mechanisms for AI SDK

Automatically handle API failures, content filtering and timeouts by switching between different AI models.

`ai-retry` wraps the provided base model with a set of retry conditions (retryables). When a request fails with an error or the response is not satisfying, it iterates through the given retryables to find a suitable fallback model. It automatically tracks which models have been tried and how many attempts have been made to prevent infinite loops.

It supports two types of retries:
- Error-based retries: when the model throws an error (e.g. timeouts, API errors, etc.)
- Result-based retries: when the model returns a successful response that needs retrying (e.g. content filtering, etc.)

### Installation

This library only supports AI SDK v5.

> [!WARNING]  
> `ai-retry` is in an early stage and the API may change in future releases.

```bash
npm install ai-retry
```

### Usage

Create a retryable model by providing a base model and a list of retryables or fallback models.

> [!INFO]  
> `ai-retry` currently supports `generateText`, `generateObject`, `streamText`, and `streamObject` calls.
> Note that streaming retry has limitations: retries are only possible before content starts flowing or very early in the stream.

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
    // Retry strategies and fallbacks...
  ],
});

// Use like any other AI SDK model
const result = await generateText({
  model: retryableModel,
  prompt: 'Hello world!',
});

// Or with streaming
const result = streamText({
  model: retryableModel,
  prompt: 'Write a story about a robot...',
});

for await (const chunk of result.textStream) {
  console.log(chunk.text);
}
```

#### Content Filter

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

#### Request Timeout

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
  abortSignal: AbortSignal.timeout(60_000), 
});
```

#### Service Overloaded

Handle service overload errors (HTTP code 529) by switching to a provider.

```typescript
import { serviceOverloaded } from 'ai-retry/retryables';

const retryableModel = createRetryable({
  model: azure('gpt-4'),
  retries: [
    serviceOverloaded(openai('gpt-4')), // Switch to OpenAI if Azure is overloaded
  ],
});
```

#### Request Not Retryable

Handle cases where the base model fails with a non-retryable error.

> [!NOTE] 
> You can check if an error is retryable with the `isRetryable` property on an [`APICallError`](https://ai-sdk.dev/docs/reference/ai-sdk-errors/ai-api-call-error#ai_apicallerror).


```typescript
import { requestNotRetryable } from 'ai-retry/retryables';

const retryable = createRetryable({
  model: azure('gpt-4-mini'),
  retries: [
    requestNotRetryable(openai('gpt-4')), // Switch provider if error is not retryable
  ],
});
```

#### Fallbacks

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

#### Custom

Create your own retryables for specific use cases:

```typescript
import { anthropic } from '@ai-sdk/anthropic';
import { openai } from '@ai-sdk/openai';
import { APICallError } from 'ai';
import { createRetryable, isErrorAttempt } from 'ai-retry';
import type { Retryable } from 'ai-retry';

const rateLimitRetry: Retryable = (context) => {
  if (isErrorAttempt(context.current)) {
    const { error } = context.current;

    if (APICallError.isInstance(error) && error.statusCode === 429) {
      return { model: anthropic('claude-3-haiku-20240307') };
    }
  }

  return undefined;
};

const retryableModel = createRetryable({
  model: openai('gpt-4'),
  retries: [
    rateLimitRetry
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

### Retryables

A retryable is a function that receives the current attempt and determines whether to retry with a different model based on the error/result and any previous attempts. 
There are several built-in retryables:

- [`contentFilterTriggered`](./src/retryables/content-filter-triggered.ts): Content filter was triggered based on the prompt or completion.
- [`requestTimeout`](./src/retryables/request-timeout.ts): Request timeout occurred.
- [`requestNotRetryable`](./src/retryables/request-not-retryable.ts): Request failed with a non-retryable error.
- [`serviceOverloaded`](./src/retryables/service-overloaded.ts): Response with status code 529 (service overloaded).

By default, each retryable will only attempt to retry once per model to avoid infinite loops. You can customize this behavior by returning a `maxAttempts` value from your retryable function.

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
