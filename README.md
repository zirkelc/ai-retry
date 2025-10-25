<div align='center'>

# ai-retry

<p align="center">Retry and fallback mechanisms for AI SDK</p>
<p align="center">
  <a href="https://www.npmjs.com/package/ai-retry" alt="ai-retry"><img src="https://img.shields.io/npm/dt/ai-retry?label=ai-retry"></a> <a href="https://github.com/zirkelc/ai-retry/actions/workflows/ci.yml" alt="CI"><img src="https://img.shields.io/github/actions/workflow/status/zirkelc/ai-retry/ci.yml?branch=main"></a>
</p>

</div>

Automatically handle API failures, content filtering, timeouts and other errors by switching between different AI models and providers.

`ai-retry` wraps the provided base model with a set of retry conditions (retryables). When a request fails with an error or the response is not satisfying, it iterates through the given retryables to find a suitable fallback model. It automatically tracks which models have been tried and how many attempts have been made to prevent infinite loops.

It supports two types of retries:
- Error-based retries: when the model throws an error (e.g. timeouts, API errors, etc.)
- Result-based retries: when the model returns a successful response that needs retrying (e.g. content filtering, etc.)

### Installation

This library only supports AI SDK v5.

```bash
npm install ai-retry
```

### Usage

Create a retryable model by providing a base model and a list of retryables or fallback models.
When an error occurs, it will evaluate each retryable in order and use the first one that indicates a retry should be attempted with a different model.

> [!NOTE]
> `ai-retry` supports both language models and embedding models.

```typescript
import { openai } from '@ai-sdk/openai';
import { generateText, streamText } from 'ai';
import { createRetryable } from 'ai-retry';

// Create a retryable model
const retryableModel = createRetryable({
  // Base model
  model: openai('gpt-4-mini'),
  retries: [
    // Retry strategies and fallbacks...
  ],
});

// Use like any other AI SDK model
const result = await generateText({
  model: retryableModel,
  prompt: 'Hello world!',
});

console.log(result.text);

// Or with streaming
const result = streamText({
  model: retryableModel,
  prompt: 'Write a story about a robot...',
});

for await (const chunk of result.textStream) {
  console.log(chunk.text);
}
```

This also works with embedding models:

```typescript
import { openai } from '@ai-sdk/openai';
import { embed } from 'ai';
import { createRetryable } from 'ai-retry';

// Create a retryable model
const retryableModel = createRetryable({
  // Base model
  model: openai.textEmbedding('text-embedding-3-large'),
  retries: [
    // Retry strategies and fallbacks...
  ],
});

// Use like any other AI SDK model
const result = await embed({
  model: retryableModel,
  value: 'Hello world!',
});

console.log(result.embedding);
```

### Retryables

The objects passed to the `retries` are called retryables and control the retry behavior. We can distinguish between two types of retryables:

- **Static retryables** are simply models instances (language or embedding) that will always be used when an error occurs. This is also called a fallback model.
- **Dynamic retryables** are functions that receive the current attempt context (error/result and previous attempts) and decide whether to retry with a different model based on custom logic.

You can think of `retries` as a big `if-else` block, where each dynamic retryable is an `if` condition that can match a certain error/result condition, and static retryables are the `else` branches that match all other conditions. The analogy is not perfect, because the order of retryables matters because `retries` are evaluated in order until one matches:

```typescript
import { openai } from '@ai-sdk/openai';
import { generateText, streamText } from 'ai';
import { createRetryable } from 'ai-retry';

const retryableModel = createRetryable({
  // Base model
  model: openai('gpt-4-mini'),
  // Retryables are evaluated in order
  retries: [
    // Dynamic retryable that matches only certain errors
    (context) => {
      return context.current.error.statusCode === 429 
        ? { model: openai('gpt-3.5-turbo') }  // Retry with this model
        : undefined;                          // Skip to next retryable
    },

    // Static retryable that always matches (fallback)
    anthropic('claude-3-haiku-20240307'),
    // Same as:
    // { model: anthropic('claude-3-haiku-20240307'), maxAttempts: 1 }
  ],
});
```

In this example, if the base model fails with a 429 error, it will retry with `gpt-4`. In any other error case, it will fallback to `gpt-3.5-turbo`. If the order would be reversed, the static retryable would catch all errors first, and the dynamic retryable would never be reached.

#### Fallbacks

If you don't need precise error matching with custom logic and just want to fallback to different models on any error, you can simply provide a list of models.

> [!NOTE] 
> Use the object syntax `{ model: openai('gpt-4') }` if you need to provide additional options like `maxAttempts`, `delay`, etc.

```typescript
import { openai } from '@ai-sdk/openai';
import { generateText, streamText } from 'ai';
import { createRetryable } from 'ai-retry';

const retryableModel = createRetryable({
  // Base model
  model: openai('gpt-4-mini'),
  // List of fallback models
  retries: [
    openai('gpt-3.5-turbo'), // Fallback for first error
    // Same as:
    // { model: openai('gpt-3.5-turbo'), maxAttempts: 1 },

    anthropic('claude-3-haiku-20240307'), // Fallback for second error
    // Same as:
    // { model: anthropic('claude-3-haiku-20240307'), maxAttempts: 1 },
  ],
});
```

In this example, if the base model fails, it will retry with `gpt-3.5-turbo`. If that also fails, it will retry with `claude-3-haiku-20240307`. If that fails again, the whole retry process stops and a `RetryError` is thrown.

#### Custom

If you need more control over when to retry and which model to use, you can create your own custom retryable. This function is called with a context object containing the current attempt (error or result) and all previous attempts and needs to return a retry model or `undefined` to skip to the next retryable. The object you return from the retryable function is the same as the one you provide in the `retries` array.

> [!NOTE]
> You can return additional options like `maxAttempts`, `delay`, etc. along with the model.

```typescript
import { anthropic } from '@ai-sdk/anthropic';
import { openai } from '@ai-sdk/openai';
import { APICallError } from 'ai';
import { createRetryable, isErrorAttempt } from 'ai-retry';
import type { Retryable } from 'ai-retry';

// Custom retryable that retries on rate limit errors (429)
const rateLimitRetry: Retryable = (context) => {
  // Only handle error attempts
  if (isErrorAttempt(context.current)) {
    // Get the error from the current attempt
    const { error } = context.current;

    // Check for rate limit error
    if (APICallError.isInstance(error) && error.statusCode === 429) {
      // Retry with a different model
      return { model: anthropic('claude-3-haiku-20240307') };
    }
  }

  // Skip to next retryable
  return undefined;
};

const retryableModel = createRetryable({
  // Base model
  model: openai('gpt-4-mini'), 
  retries: [
    // Use custom rate limit retryable
    rateLimitRetry

    // Other retryables...
  ],
});
```

In this example, if the base model fails with a 429 error, it will retry with `claude-3-haiku-20240307`. For any other error, it will skip to the next retryable (if any) or throw the original error.

#### All Retries Failed

If all retry attempts failed, a `RetryError` is thrown containing all individual errors.
If no retry was attempted (e.g. because all retryables returned `undefined`), the original error is thrown directly.

```typescript
import { RetryError } from 'ai';

const retryableModel = createRetryable({
  // Base model = first attempt
  model: azure('gpt-4-mini'), 
  retries: [
    // Fallback model 1 = Second attempt
    openai('gpt-3.5-turbo'), 
    // Fallback model 2 = Third attempt
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

Errors are tracked per unique model (provider + modelId). That means on the first error, it will retry with `gpt-3.5-turbo`. If that also fails, it will retry with `claude-3-haiku-20240307`. If that fails again, the whole retry process stops and a `RetryError` is thrown.

### Built-in Retryables

There are several built-in dynamic retryables available for common use cases:

- [`contentFilterTriggered`](./src/retryables/content-filter-triggered.ts): Content filter was triggered based on the prompt or completion.
- [`requestTimeout`](./src/retryables/request-timeout.ts): Request timeout occurred.
- [`requestNotRetryable`](./src/retryables/request-not-retryable.ts): Request failed with a non-retryable error.
- [`retryAfterDelay`](./src/retryables/retry-after-delay.ts): Retry with delay and exponential backoff and respect `retry-after` headers.
- [`serviceOverloaded`](./src/retryables/service-overloaded.ts): Response with status code 529 (service overloaded).
  - Use this retryable to handle Anthropic's overloaded errors.

#### Content Filter

Automatically switch to a different model when content filtering blocks your request.

> [!WARNING]  
> This retryable currently does not work with streaming requests, because the content filter is only indicated in the final response.

```typescript
import { contentFilterTriggered } from 'ai-retry/retryables';

const retryableModel = createRetryable({
  model: azure('gpt-4-mini'),
  retries: [
    contentFilterTriggered(openai('gpt-4-mini')), // Try OpenAI if Azure filters
  ],
});
```

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

Handle service overload errors (status code 529) by switching to a provider.

> [!NOTE] 
> You can use this retryable to handle Anthropic's overloaded errors.

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

#### Retry After Delay

If an error is retryable, such as 429 (Too Many Requests) or 503 (Service Unavailable) errors, it will be retried after a delay. 
The delay and exponential backoff can be configured. If the response contains a [`retry-after`](https://developer.mozilla.org/en-US/docs/Web/HTTP/Reference/Headers/Retry-After) header, it will be prioritized over the configured delay.

Note that this retryable does not accept a model parameter, it will always retry the model from the latest failed attempt.

```typescript
import { retryAfterDelay } from 'ai-retry/retryables';

const retryableModel = createRetryable({
  model: openai('gpt-4'), // Base model
  retries: [
    // Retry base model 3 times with fixed 2s delay
    retryAfterDelay({ delay: 2000, maxAttempts: 3 }),

    // Or retry with exponential backoff (2s, 4s, 8s)
    retryAfterDelay({ delay: 2000, backoffFactor: 2, maxAttempts: 3 }),

    // Or retry only if the response contains a retry-after header
    retryAfterDelay({ maxAttempts: 3 }),
  ],
});
```

By default, if a [`retry-after-ms`](https://learn.microsoft.com/en-us/azure/ai-foundry/openai/how-to/provisioned-get-started#what-should--i-do-when-i-receive-a-429-response) or [`retry-after`](https://developer.mozilla.org/en-US/docs/Web/HTTP/Reference/Headers/Retry-After) header is present in the response, it will be prioritized over the configured delay. The delay from the header will be capped at 60 seconds for safety.

### Options

#### Retry Delays

You can delay retries with an optional exponential backoff. The delay respects abort signals, so requests can still be cancelled during the delay period.

```typescript
const retryableModel = createRetryable({
  model: openai('gpt-4'),
  retries: [
    // Retry model 3 times with fixed 2s delay
    { model: openai('gpt-4'), delay: 2000, maxAttempts: 3 },

    // Or retry with exponential backoff (2s, 4s, 8s)
    { model: openai('gpt-4'), delay: 2000, backoffFactor: 2, maxAttempts: 3 },
  ],
});

const result = await generateText({
  model: retryableModel,
  prompt: 'Write a vegetarian lasagna recipe for 4 people.',
  // Will be respected during delays
  abortSignal: AbortSignal.timeout(60_000), 
});
```

You can also use delays with built-in retryables:

```typescript
import { serviceOverloaded } from 'ai-retry/retryables';

const retryableModel = createRetryable({
  model: openai('gpt-4'),
  retries: [
    // Wait 5 seconds before retrying on service overload
    serviceOverloaded(openai('gpt-4'), { maxAttempts: 3, delay: 5_000 }),
  ],
});
```

#### Max Attempts

By default, each retryable will only attempt to retry once per model to avoid infinite loops. You can customize this behavior by returning a `maxAttempts` value from your retryable function. Note that the initial request with the base model is counted as the first attempt.

```typescript
const retryableModel = createRetryable({
  model: openai('gpt-4'),
  retries: [
    // Try this once
    anthropic('claude-3-haiku-20240307'), 
    // Try this one more time (initial + 1 retry)
    { model: openai('gpt-4'), maxAttempts: 2 }, 
    // Already tried, won't be retried again
    anthropic('claude-3-haiku-20240307') 
  ],
});
```

The attempts are counted per unique model (provider + modelId). That means if multiple retryables return the same model, it won't be retried again once the `maxAttempts` is reached.

#### Provider Options

You can override provider-specific options for each retry attempt. This is useful when you want to use different configurations for fallback models.

```typescript
const retryableModel = createRetryable({
  model: openai('gpt-5'),
  retries: [
    // Use different provider options for the retry
    {
      model: openai('gpt-4o-2024-08-06'),
      providerOptions: {
        openai: {
          user: 'fallback-user',
          structuredOutputs: false,
        },
      },
    },
  ],
});

// Original provider options are used for the first attempt
const result = await generateText({
  model: retryableModel,
  prompt: 'Write a story',
  providerOptions: {
    openai: {
      user: 'primary-user',
    },
  },
});
```

The retry's `providerOptions` will completely replace the original ones during retry attempts. This works for all model types (language and embedding) and all operations (generate, stream, embed).

#### Logging

You can use the following callbacks to log retry attempts and errors:
- `onError` is invoked if an error occurs. 
- `onRetry` is invoked before attempting a retry.

```typescript
const retryableModel = createRetryable({
  model: openai('gpt-4-mini'),
  retries: [/* your retryables */],
  onError: (context) => {
    console.error(`Attempt ${context.attempts.length} with ${context.current.model.provider}/${context.current.model.modelId} failed:`, 
      context.current.error
    );
  },
  onRetry: (context) => {
    console.log(`Retrying attempt ${context.attempts.length + 1} with model ${context.current.model.provider}/${context.current.model.modelId}...`);
  },
});
```

### Streaming

Errors during streaming requests can occur in two ways:

1. When the stream is initially created (e.g. network error, API error, etc.) by calling `streamText`.
2. While the stream is being processed (e.g. timeout, API error, etc.) by reading from the returned `result.textStream` async iterable.

In the second case, errors during stream processing will not always be retried, because the stream might have already emitted some actual content and the consumer might have processed it. Retrying will be stopped as soon as the first content chunk (e.g. types of `text-delta`, `tool-call`, etc.) is emitted. The type of chunks considered as content are the same as the ones that are passed to [onChunk()](https://github.com/vercel/ai/blob/1fe4bd4144bff927f5319d9d206e782a73979ccb/packages/ai/src/generate-text/stream-text.ts#L684-L697).

### API Reference

#### `createRetryable(options: RetryableModelOptions): LanguageModelV2 | EmbeddingModelV2`

Creates a retryable model that works with both language models and embedding models.

```ts
interface RetryableModelOptions<MODEL extends LanguageModelV2 | EmbeddingModelV2> {
  model: MODEL;
  retries: Array<Retryable<MODEL> | MODEL>;
  onError?: (context: RetryContext<MODEL>) => void;
  onRetry?: (context: RetryContext<MODEL>) => void;
}
```

#### `Retryable`

A `Retryable` is a function that receives a `RetryContext` with the current error or result and model and all previous attempts.
It should evaluate the error/result and decide whether to retry by returning a `Retry` or to skip by returning `undefined`.

```ts
type Retryable = (
  context: RetryContext
) => Retry | Promise<Retry> | undefined;
```

#### `Retry`

A `Retry` specifies the model to retry and optional settings like `maxAttempts`, `delay`, `backoffFactor`, and `providerOptions`.

```typescript
interface Retry {
  model: LanguageModelV2 | EmbeddingModelV2;
  maxAttempts?: number;      // Maximum retry attempts per model (default: 1)
  delay?: number;            // Delay in milliseconds before retrying
  backoffFactor?: number;    // Multiplier for exponential backoff
  providerOptions?: ProviderOptions; // Provider-specific options for the retry
}
```

**Options:**
- `model`: The model to use for the retry attempt.
- `maxAttempts`: Maximum number of times this model can be retried. Default is 1.
- `delay`: Delay in milliseconds to wait before retrying. The delay respects abort signals from the request.
- `backoffFactor`: Multiplier for exponential backoff (`delay × backoffFactor^attempt`). If not provided, uses fixed delay.
- `providerOptions`: Provider-specific options that override the original request's provider options during retry attempts.

#### `RetryContext`

The `RetryContext` object contains information about the current attempt and all previous attempts.

```typescript
interface RetryContext {
  current: RetryAttempt;
  attempts: Array<RetryAttempt>;
}
```

#### `RetryAttempt`

A `RetryAttempt` represents a single attempt with a specific model, which can be either an error or a successful result that triggered a retry.

```typescript
// For both language and embedding models
type RetryAttempt =
  | { type: 'error'; error: unknown; model: LanguageModelV2 | EmbeddingModelV2 }
  | { type: 'result'; result: LanguageModelV2Generate; model: LanguageModelV2 };

// Note: Result-based retries only apply to language models, not embedding models

// Type guards for discriminating attempts
function isErrorAttempt(attempt: RetryAttempt): attempt is RetryErrorAttempt;
function isResultAttempt(attempt: RetryAttempt): attempt is RetryResultAttempt;
```

### License

MIT
