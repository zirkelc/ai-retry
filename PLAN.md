# Call-level retry API — design plan

Status: **implemented** (§1–§14). The five functions ship from the root `ai-retry` barrel; `src/call/`
holds the loop, the shared machinery and one module per entry point. The design probes have been
replaced by
a `.test-d.ts` beside each entry point, which makes the same claims against the real functions.

This document is kept as the design record: what was measured, what was rejected, and why. Where
implementation forced a change, the section says so — see [§13](#13-what-changed-during-implementation)
for the list.

Environment this was designed against and built on: `ai@7.0.35`, `@ai-sdk/provider@4.0.3`,
TypeScript 5.9, vitest 4 with `typecheck.enabled: true`.

---

## 1. Objective

Replace `createRetryableModel` with a set of **call-level** retry functions that wrap the AI SDK
entry points directly, so that the _complete_ retry logic lives above the call rather than below
the model.

`createRetryableModel` retries inside `doGenerate`/`doStream`, i.e. **below** the entry point.
That makes it structurally blind to any failure that lives _on_ the call:

- a `streamText` timeout (`timeout.firstChunkMs` / `stepMs` / `totalMs`)
- an inbound `abortSignal` deadline

Once those fire, the SDK tears the call down and discards whatever a lower retry produced
(issue [#50](https://github.com/zirkelc/ai-retry/issues/50)). Re-running the **whole call** with the
next model is the only way to recover them.

The new API must be able to do everything `createRetryableModel` does, so users can migrate
without losing capability — this is why result-based retries are in scope (§7).

### Non-goals

- Keeping `createRetryableModel` API-compatible. It stays **as-is, untouched**, and is _not_
  reused by the new API. A future release may deprecate it; that is out of scope here.
- Supporting arbitrary user functions. The new API wraps five known entry points only.

---

## 2. Scope

### In scope — five entry points

| entry point     | model family | returns         | generics                             |
| --------------- | ------------ | --------------- | ------------------------------------ |
| `generateText`  | Language     | `Promise`       | `TOOLS`, `RUNTIME_CONTEXT`, `OUTPUT` |
| `streamText`    | Language     | **synchronous** | `TOOLS`, `RUNTIME_CONTEXT`, `OUTPUT` |
| `embed`         | Embedding    | `Promise`       | none                                 |
| `embedMany`     | Embedding    | `Promise`       | none                                 |
| `generateImage` | Image        | `Promise`       | none                                 |

### Out of scope

- **`generateObject` / `streamObject`** — explicitly excluded by the user.
- **`transcribe` / `generateSpeech`** — undecided, see [Open questions](#12-open-questions).
- **Sticky models (`reset`)** — parked; see §9.4 for why the merged-args design has nowhere to
  put the state.
- **A generic `wrapRetryable(fn, …)` escape hatch** — deleted, see §10.

---

## 3. Public API

Five functions, each taking a single merged argument object: the entry point's own arguments
**plus** a `retry` field. `model` stays a normal call argument and is switched per attempt.

```ts
import { retryableStreamText } from 'ai-retry';

const result = await retryableStreamText({
  model: openai('gpt-4o'),
  prompt: 'Invent a new holiday.',
  timeout: { firstChunkMs: 2_000 },
  retry: [
    serviceOverloaded(fallbackModel),
    timeout().switch({ model: fastModel, timeout: 1_000 }),
  ],
});

for await (const chunk of result.textStream) process.stdout.write(chunk);
```

### 3.1 The `retry` argument

```ts
type RetryArg<MODEL, INPUT> =
  | Retries<MODEL, INPUT> // bare array — the documented common form
  | {
      retries: Retries<MODEL, INPUT>;
      disabled?: boolean | (() => boolean);
      telemetry?: RetryTelemetrySettings;
      onError?: (context: RetryContext<MODEL>) => void;
      onRetry?: (
        context: RetryContext<MODEL>,
      ) =>
        | void
        | OnRetryOverrides<MODEL, INPUT>
        | Promise<void | OnRetryOverrides<MODEL, INPUT>>;
      onSuccess?: (context: SuccessContext<MODEL>) => void;
      onFailure?: (context: FailureContext<MODEL>) => void;
    };
```

Single namespaced `retry` key rather than loose top-level `retries` / `onRetry` / `disabled`
fields, so exactly **one** name is in collision range if the SDK adds arguments later. The bare
array shorthand is the primary documented form; the object form is the escape hatch for hooks.

### 3.2 Signatures

Two shapes, because only `streamText` and `generateText` are generic.

**Tool-generic entry points — hand-written, carries an override list:**

```ts
type RetryableStreamText = <
  TOOLS extends ToolSet,
  INPUT extends StreamTextInput = never,
>(
  args: Omit<Parameters<typeof streamText>[0], 'tools' | 'activeTools'> & {
    tools?: TOOLS;
    activeTools?: Array<keyof TOOLS & string>;
    retry?: RetryArg<LanguageModel, INPUT>;
  },
) => Promise<ReturnType<typeof streamText<TOOLS>>>;
```

**Non-generic entry points — free, nothing to drift:**

```ts
type RetryableEmbed = <INPUT extends EmbedInput = never>(
  args: Parameters<typeof embed>[0] & {
    retry?: RetryArg<EmbeddingModel, INPUT>;
  },
) => ReturnType<typeof embed>;
```

`retryableStreamText` returns a **`Promise`** even though `streamText` is synchronous — the loop
must know which attempt won before it can hand a result back. This is the only place the wrapped
signature differs from the original. **The user accepted this explicitly ("async is fine").**

`retryableGenerateText` returns `Promise` too, which matches `generateText` exactly.

---

## 4. Type-system findings (all measured, not assumed)

This section records _why_ the signatures look the way they do. Every claim below was verified
with a `.test-d.ts` probe run under `vitest --typecheck`.

### 4.1 Deriving args from `Parameters<FN>[0]` destroys generic inference

`Parameters<FN>` instantiates a generic function at its constraints/defaults, yielding a
**monomorphic** signature. Measured consequences:

- `generateObject`'s `schema` no longer types `result.object`
- `activeTools: ['nope']` stops being rejected

TypeScript has no higher-kinded types, so _"the same polymorphic signature with its return type
mapped through `Promise`"_ is inexpressible. You can pass a generic function type through
untouched, or destructure it into concrete pieces — nothing in between.

### 4.2 Approaches rejected

| approach                                                                                   | result                                                                                                                                                                                  |
| ------------------------------------------------------------------------------------------ | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `Omit<Parameters<FN>[0], 'model'>` + `Promise<ReturnType<FN>>`                             | inference collapses (4.1)                                                                                                                                                               |
| plain `Omit` over `generateObject`'s args                                                  | **drops `schema` entirely** — its options are a _union_ over output modes and `Omit` keeps only common keys. Needs a `DistributiveOmit`                                                 |
| instantiation expression `typeof streamText<TOOLS>` in _argument_ position                 | TS cannot infer `TOOLS` _through_ `Parameters<typeof f<T>>`; falls back to the constraint                                                                                               |
| re-instantiating the result from inferred args (`ToolsOf<ARGS>`)                           | `ToolsOf` extracts correctly, but rebuilding `StreamTextResult` needs `ai`'s **non-exported** `Output` type (`Output` is exported as a _namespace_, not a type) → degrades to `unknown` |
| identity signature `wrapRetryable(fn): FN`                                                 | **works** — proven by `expectTypeOf(wrapped).toEqualTypeOf<typeof generateText>()`. But requires returning synchronously, so `streamText` would need a facade (§4.5)                    |
| second positional param `(args, retryOpts)` via overload intersection `FN & ((a, r) => R)` | 1-arg calls stay sharp, **2-arg calls silently collapse** — the failure lands exactly where you'd use per-call options                                                                  |
| curried `.with(opts)(args)` returning `FN`                                                 | **works**, both forms sharp. Parked — superseded by merged args                                                                                                                         |

### 4.3 What works: hand-written signature with a real inference site

The distinction that unlocked everything:

- **Inferring** `TOOLS` _through_ `Parameters<typeof streamText<TOOLS>>` — impossible.
- **Using** `TOOLS` in _return_ position via `ReturnType<typeof streamText<TOOLS>>` — fine.

So: put `TOOLS` in a genuine inference site (`tools?: TOOLS`), take the remaining arguments from
`Parameters<typeof streamText>[0]` minus the overridden fields, and use the instantiation
expression only for the return type.

Measured: the wrapped result type is **identical to a direct call's, member for member**
(`toolResults`, `toolCalls`, `steps`, `content`), and `activeTools` narrowing is restored.

> **Methodology note.** Assert against a _direct call_ (`toEqualTypeOf<Awaited<typeof direct.x>>`),
> never against a literal type. An early probe "failed" claiming lost inference when in fact the
> SDK itself types `streamText().toolResults[0].output` as `unknown`. The direct-call baseline is
> what distinguishes "we lost it" from "it was never there".

### 4.4 Merged args work — the earlier blocker was the derivation, not the merge

Adding fields to a hand-written args type costs nothing. Measured for **both** `generateText` and
`streamText`:

- result types identical to a direct call
- `activeTools` still narrowed
- **excess property checking survives the intersection** — `nonsense: true` still rejected
- `retry` optional; both the bare-array and object forms preserve inference

Non-generic entry points get this for free: `Parameters<typeof embed>[0] & RetryArgs` is exact,
with **no override list and nothing to drift**. Verified that an `ImageModel` is rejected as an
embedding fallback.

**Cost of merging:** `generateText` loses its free ride. With a factory form
(`retryableGenerateText(opts): typeof generateText`) it needed _no_ hand-written signature at all.
Merging means hand-writing both language signatures, so there are two override lists instead of
one. Accepted in exchange for one call step and per-call retry options.

### 4.5 The facade — designed, then dropped

To keep `streamText` synchronous (a true drop-in), the wrapper would return an object satisfying
`StreamTextResult` backed by a pending `Promise<StreamTextResult>`. `StreamTextResult` in 7.0.35 is
23 `PromiseLike` members, 6 stream members, 6 methods:

- promise members → `resultPromise.then(r => r.text)` — trivial
- stream members → one `deferredStream()` helper using a `pull`-based `ReadableStream` (pull is
  only invoked when the consumer wants a chunk, so back-pressure is preserved and the `await`
  inside `pull` does the deferring)
- `consumeStream` / `pipe*ToResponse` → `async (...a) => (await p)[name](...a)`
- `toUIMessageStreamResponse()` / `toTextStreamResponse()` → must return a `Response`
  **synchronously**, so headers would have to be reproduced by hand. Mitigating discovery: the
  UI-message variants are **already deprecated** in 7.0.35 in favour of standalone helpers over
  `result.stream`
- a generic `Proxy` **cannot** work: `get` must decide synchronously whether a member is a promise,
  a stream, or a method, and nothing distinguishes them before the result exists

**Dropped** because §4.3 showed a hand-written signature recovers full inference _without_ it. The
facade's only remaining value was the synchronous call shape, which the user waived. It would also
have converted the `ai` relationship from types-only to structural (~150 lines pinned to the result
interface).

### 4.6 The `INPUT` generic — not the `MODEL` trap

`Retry.options` must be **entry-point-scoped**, not family-scoped: `embed` takes `value`,
`embedMany` takes `values`. (`EmbeddingModelRetryCallOptions` today picks `values` — the
provider-level `doEmbed` shape — which is wrong for `embed` at the call layer.)

The concern was that an `INPUT` generic would repeat the `MODEL` problem, where nothing at
construction time can infer the parameter, forcing family-bound exports
(`ai-retry/language-model/conditions` etc.).

**It does not.** `MODEL` has no inference source — the model arrives in a later `.switch()` call.
`INPUT` is inferable from the `options` literal you write. Measured:

| case                                                | result                                                        |
| --------------------------------------------------- | ------------------------------------------------------------- |
| `.switch({ model })`, no options                    | works, and stays **portable** — accepted by every entry point |
| `.switch({ model, options: { value } })` → `embed`  | accepted                                                      |
| `.switch({ model, options: { values } })` → `embed` | correctly **rejected**                                        |
| inline `retry: [{ model, options: {…} }]`           | contextual typing, no generic involved                        |
| unbound `error(…)` without a family-bound export    | still needs `MODEL` pinned — trap confirmed distinct          |

Three constraints on the implementation:

1. **`INPUT` MUST be constrained** to the entry point's override type
   (`INPUT extends StreamTextInput = never`). Measured: an _unconstrained_ `INPUT` is a black hole
   that infers whatever you write and rejects nothing.
2. **The `never` default is load-bearing.** `options?: never` is assignable to any target, which is
   what keeps no-option retryables portable. With `unknown` nothing would be assignable and every
   existing retryable would break.
3. **`options` must stay in return position.** `Retryable` returns `Retry`, so it is covariant. If
   `options` ever appeared in an input position (e.g. a callback receiving resolved options), the
   variance flips and this breaks.

Adding a defaulted `INPUT` to `Retry` / `Retryable` / `Condition` is **non-breaking** for the model
API — `Retry<LanguageModel>` keeps compiling — so conditions and retryables serve both APIs.

`TOOLS` and `INPUT` were verified to coexist in one args object without interfering.

---

## 5. Architecture

Three layers. Only the middle one has five of anything.

```
retryableGenerateText  retryableStreamText  retryableEmbed  retryableEmbedMany  retryableGenerateImage
        └──────────────────────┬──────────────────────────────────────────────┘
                    runRetryLoop({ entryPoint, args, policy })          ← all retry logic, once
                          ├─ resolveModel()          gateway string → instance
                          ├─ findRetryModel()        conditions + attempt counting
                          ├─ evaluateError()
                          ├─ resolveBackoffDelay()
                          ├─ prepareRetryError()
                          └─ createRetryTelemetry()
                    EntryPoint table                                     ← the only variation
```

### 5.1 The loop (internal, not exported)

```ts
async function runRetryLoop<MODEL, ARGS, RESULT, INPUT>(input: {
  entryPoint: EntryPoint<MODEL, ARGS, RESULT>;
  args: ARGS;
  policy: RetryPolicy<MODEL, INPUT>;
}): Promise<RESULT> {
  const base = resolveModel(args.model);
  let current = base, currentRetry, attempts = [];

  while (true) {
    const overrides = (await policy.onRetry?.(context)) ?? currentRetry?.options;

    const attemptArgs = entryPoint.deadline(
      { ...args, ...overrides, model: current, maxRetries: args.maxRetries ?? 0 },
      currentRetry?.timeout ?? callDeadlineOf(args),
      args.abortSignal,                      // the caller's RAW signal
    );

    try {
      const result = await entryPoint.call(attemptArgs);
      const settled = await entryPoint.settle?.(result, attempt) ?? { type: 'committed' };

      if (settled.type === 'result') {
        const next = await findRetryModel(policy.retries, resultContext(settled.result));
        if (next) { current = next.model; currentRetry = next; continue; }
      }

      policy.onSuccess?.(…);
      return result;
    } catch (error) {
      const next = await evaluateError({ error, model: current, attempts, retries: policy.retries });
      if (!next || args.abortSignal?.aborted) throw prepareRetryError(attempts, error);
      await delay(resolveBackoffDelay(next, attempts));
      current = next.model; currentRetry = next;
    }
  }
}
```

**Key property:** `settle` throwing is indistinguishable from `call` throwing. A stream that emits
an `error` part before content, or trips a deadline, throws out of `settle` and lands in the same
`catch`. Commit detection therefore reuses the entire error path with **no branch in the loop**.

### 5.2 The entry-point table

```ts
type EntryPoint<MODEL, ARGS, RESULT> = {
  operation: string; // telemetry span name
  call: (args: ARGS) => Promise<RESULT>;
  deadline: DeadlineStrategy<ARGS>; // two implementations, shared
  settle?: (result: RESULT, attempt: Attempt<MODEL>) => Promise<Settled>;
};

type Settled =
  | { type: 'committed' } // terminal, cannot fail over
  | { type: 'result'; result: CallResult }; // judge it against result conditions
```

```ts
const generateTextEntry = {
  operation: 'generateText',
  call: generateText,
  deadline: viaTimeoutArg,
  settle: async (r) => ({ type: 'result', result: toCallResult(r) }),
};
const streamTextEntry = {
  operation: 'streamText',
  deadline: viaTimeoutArg,
  call: async (a) => streamText({ onError: NOOP, ...a }),
  settle: (r, at) => detectStreamCommit(r.stream, at),
};
const embedEntry = {
  operation: 'embed',
  call: embed,
  deadline: viaAbortSignal,
};
const embedManyEntry = {
  operation: 'embedMany',
  call: embedMany,
  deadline: viaAbortSignal,
};
const generateImageEntry = {
  operation: 'generateImage',
  call: generateImage,
  deadline: viaAbortSignal,
};
```

Five rows, two deadline strategies, two `settle` implementations. The loop never names `timeout`
or `abortSignal` — that knowledge lives entirely in the strategies.

> **Changed in review.** The rows are **not** one table in one file. Each lives in its own module
> under `call/functions/` (`generate-text.ts`, `stream-text.ts`, `embed.ts`, `embed-many.ts`,
> `generate-image.ts`) next to that entry point's public signature and export, so everything
> specific to `streamText` reads in one place. `call/retryable-calls.ts` holds only what is
> genuinely shared: the two deadline strategies and `defineRetryableCall`.
>
> The cost, accepted deliberately: the five rows can no longer be compared at a glance, which is how
> the "two strategies, two `settle`s" shape used to be self-evident. This section is now the place
> that records it.
>
> Two simplifications fell out of the move, both verified rather than assumed:
>
> - **`defineEntryPoint` is gone.** It existed to type the row separately from the function built
>   from it. With one file per entry point nothing ever holds a row on its own, so it merged into
>   `defineRetryableCall`, which now takes the row directly.
> - **The `RESULT` erasure is gone.** It existed because the exported entry consts leaked the SDK's
>   non-exportable `Output` type into declaration emit (TS4023). The rows are no longer exported —
>   the only export is the function, whose declared type is the explicit `as RetryableXxx` cast — so
>   nothing infers through to `Output`. `pnpm build` (with attw and publint) is what confirms this;
>   if a future change exports a row again, TS4023 returns and the erasure has to come back.

**Decision history:** the adapter was proposed, then argued _against_ when scope was two language
entry points (with only `generateText`/`streamText`, `deadline` is identical for both and the
adapter degenerates to a parameter list), then reinstated when `embed`/`embedMany`/`generateImage`
came into scope — those split the deadline axis genuinely. Do not remove it again without
re-checking §6.1.

### 5.3 Public functions

Each entry point's module ends with this, so the cast sits beside the signature it casts to.

```ts
const define = (entryPoint) => (args) => {
  const { retry, ...callArgs } = args;
  return runRetryLoop({ entryPoint, args: callArgs, policy: toPolicy(retry) });
};

export const retryableEmbed = define(embedEntry) as RetryableEmbed;
```

One cast per entry point — the implementation cannot wear the hand-written generic signature.
Localised to one line each; the drift tests sit on the public type.

---

## 6. Per-attempt argument handling

### 6.1 Deadlines — the two strategies

Verified against `ai@7.0.35`: `RequestOptions` (shared by `generateText` and `streamText`) carries
`timeout?: TimeoutConfiguration<TOOLS>`. `generateObject`/`streamObject` use
`Omit<RequestOptions, 'timeout'>`. **`embed`, `embedMany` and `generateImage` have no `timeout`
argument at all** — only `abortSignal`.

```ts
const viaTimeoutArg = (args, ms) =>
  ms === undefined
    ? args
    : { ...args, timeout: mergeTimeout(args.timeout, ms) };

const viaAbortSignal = (args, ms, caller) =>
  ms === undefined
    ? args
    : {
        ...args,
        abortSignal: AbortSignal.any([
          ...(caller ? [caller] : []),
          AbortSignal.timeout(ms),
        ]),
      };
```

`mergeTimeout` replaces `totalMs` while preserving any finer-grained windows the caller set
(`firstChunkMs`, `chunkMs`, …).

> **This was a real bug in the discarded generic wrapper.** It set `timeout` blindly on every entry
> point. Measured against a 1000 ms model with `timeout: 100`: `generateText` aborted at 113 ms,
> `generateObject` **ran to completion at 1018 ms**. A generic wrapper cannot fix this — the
> obvious workaround (always compose into `abortSignal`) is wrong for `streamText`, where an
> inbound `abortSignal` is deliberately a hard caller-cancel that must **not** fail over.

> **Critical invariant.** The loop must keep the caller's **raw** signal for the
> "already cancelled → do not fail over" check, separate from the composed per-attempt signal. If
> they are conflated, our own deadline looks like a caller cancel and silently kills fail-over.
> This is now the deadline mechanism for three of five entry points.

### 6.2 `maxRetries` defaults to `0`

The SDK's own in-call retries are disabled unless the caller sets `maxRetries` explicitly.
Otherwise the entry point would re-issue the failing model several times before the loop ever saw
the error, multiplying every deadline. Note the SDK only retries errors it considers _retryable_
(a plain `Error` is not; an `APICallError` with `isRetryable: true` is).

### 6.3 `onError` is silenced for `streamText` only

`streamText` reports stream failures to `onError` instead of throwing, and defaults it to
`console.error` — which would log every attempt the loop successfully recovered. A caller-supplied
`onError` still wins. `generateText` has no `onError` argument, so it must **not** be injected
there (the discarded generic wrapper did, landing it in `...settings`).

### 6.4 `Retry.options` merging

Per-field precedence, highest first: `onRetry` return value → `Retry.options` → the call's own
args. At the call layer `options` is a `Partial<>` of the entry point's arguments, so
**`prompt`/`messages`/`system` become legitimately overridable** — new capability relative to the
model layer, where `Retry.options.prompt` is a provider-shaped message array.

> Do **not** carry over the `@deprecated` marker that was added to `Retry.options.prompt` on the
> discarded branch. It is wrong for the model API, where a provider-shaped prompt override works
> correctly.

---

## 7. Result-based retries — in scope

### 7.1 Only `generateText` / `streamText` need anything

| family    | result-based conditions                                                                            | work      |
| --------- | -------------------------------------------------------------------------------------------------- | --------- |
| embedding | none exist (`embedding-model/conditions` exports only `error`, `httpStatus`, `timeout`, `aborted`) | **none**  |
| image     | `noImage()` is **error**-based — it matches `NoImageGeneratedError` on an _error_ attempt          | **none**  |
| language  | `result`, `finishReason`, `schemaInvalid`                                                          | all of it |

### 7.2 `generateText`

The result is complete when the call resolves. Build a result attempt, call `findRetryModel`,
retry or return. `findRetryModel` **already implements** the rule that result attempts only match
_function_ retryables (bare models and static `Retry` objects are skipped). ~20 lines.

### 7.3 `streamText` — the pre-commit window

`detectStreamCommit` returns an outcome instead of resolving void:

- a content part is seen → `{ type: 'committed' }`
- end of stream with **no** content → `{ type: 'result', result }`, with `finishReason` / `usage` /
  `providerMetadata` read straight from the `finish` part it already walks past (no need to await
  the result object's promises)

This window is exactly right: `contentFilterStreamChunks` in the test utils is
`stream-start`, `response-metadata`, `finish(content-filter)` — **no content parts**.

`isStreamContentPart` counts `text-delta`, `reasoning-delta`, `source`, `tool-call`, `tool-result`,
`tool-input-start`, `tool-input-delta`, `raw` as content.

**Inherent limitation:** pre-commit there is no text and no tool calls _by definition_ — they would
have committed the attempt. So for streams, result-based conditions are effectively
finishReason-shaped. This is **not** a regression: the model API has the same ceiling (its own test
asserts `should NOT retry when content was already streamed before content-filter finish`).

### 7.4 The condition payload — normalize, do not add a `RESULT` generic

Provider-level is `res.finishReason.unified`; SDK-level is a flat `finishReason` plus a separate
`rawFinishReason`. Existing conditions read the nested path.

**Decision: put a normalized `finishReason` (the unified string) on the attempt.** Then
`finishReason('content-filter')` reads `ctx.current.finishReason` and works unchanged in **both**
APIs.

Rejected alternative: a third generic `RESULT` on `Condition`/`Retryable`/`RetryContext` plus
parallel condition exports — expensive, and it reintroduces the `MODEL` trap.

`result(predicate)` — the raw escape hatch — is the only genuinely API-specific one, since it hands
over the whole result. See [Open questions](#12-open-questions).

### 7.5 `schemaInvalid()` — mark deprecated

It reads `responseFormat` off provider call options, which do not exist at the call layer, and with
`generateObject` out of scope there is no equivalent. **Mark `@deprecated`**; it keeps working for
the model API.

### 7.6 What this unlocks

`onSuccess` with a result. The discarded call driver had `onComplete`-without-a-result precisely
_because_ it never inspected results. Once it does, the positive hook can carry the result for
`generateText` and the commit-or-contentless-finish outcome for `streamText`.

---

## 8. `streamObject` is not supported — and why

Discovered by probe, and it invalidates a claim currently in the README and JSDoc.

`streamText`'s `stream` / `fullStream` getters **tee** per access (`ai/dist/index.js:9890`), so
reading them for commit detection is safe. `streamObject`'s `fullStream` returns
`createAsyncIterableStream(this.baseStream)` — the **base stream itself** (`index.js:13254`).
Reading it locks it; the caller's `partialObjectStream` then yields **zero partials** and throws
`ERR_INVALID_STATE`.

The existing `createRetryableStream` tests only ever fed synthetic tee-able objects, so this was
never caught.

Moot for the new API (object functions are out of scope), but if `createRetryableStream` survives
until the new API ships, its docs should be narrowed. The facade (§4.5) would have fixed it by
owning the stream members and replaying buffered parts.

---

## 9. Rejected and parked designs

### 9.1 `wrapRetryable(fn, options)` — deleted

A generic wrapper taking the entry-point function as an argument. Rationale for the reference was
"no runtime dependency on `ai`" — **this was wrong**: `ai` is a required peer dependency and is
already imported at runtime in 12 source files (`gateway`, `RetryError`, `APICallError`,
`NoImageGeneratedError`). Passing the reference only kept one file uncoupled.

Fatal flaw: a generic wrapper rewrites args blindly and cannot know that `generateObject` has no
`timeout` (§6.1). Dedicated functions also give better typing (§4.3).

### 9.2 Factory form — superseded

`retryableStreamText(options)` returning `typeof streamText`. Its advantage was that
`generateText` needed no hand-written signature at all. Superseded by merged args (§4.4), which
removes the two-step call and makes retry options per-call.

A factory can be re-added **on top** later — presetting defaults and returning the same merged-args
function — without breaking anything. The reverse is not true.

### 9.3 Per-call options via a second positional parameter — rejected

Measured to silently collapse inference for exactly the calls that use it (§4.2). The curried
`.with(opts)(args)` form works and is parked; merged args made it unnecessary.

### 9.4 Sticky models / `reset` — parked

`reset: 'after-N-requests'` requires memory **across** calls, and merged args make each call
self-contained. The principled split, if it is ever wanted:

- stateless options (`retries`, `disabled`, hooks, telemetry) → per call
- stateful options (`reset`) → require a factory that owns the state

The default `reset: 'after-request'` _is_ stateless, so the bare function covers it. Parked at the
user's request; **not** a design constraint on v1.

> A related change was made to `BaseRetryableModel` on the discarded branch (keying sticky state by
> model, because a shared driver served many base models). **Revert it.** With the model API kept
> as-is, each wrapper instance has exactly one base model and the change is unnecessary complexity.

---

## 10. What is deleted, kept, reverted

### Deleted

- `wrapRetryable` (never released)
- No escape hatch survives for non-SDK calls or user-defined wrappers. Accepted cost of a smaller
  surface; `runRetryLoop` can be exported later if anyone asks.

### Deprecated, not deleted — changed during implementation

`createRetryableCall` and `createRetryableStream` (the released `ai-retry/experimental/call` and
`ai-retry/experimental/stream` subpaths) were to be removed **in the same release** that adds the new
functions, so there was never a window with two competing call-level APIs.

**Kept instead**, marked `@deprecated`, so the release does not force anyone to migrate on the same
day it ships. The window of two call-level APIs is the price; the README resolves it by documenting
the new functions as the call-level API and the drivers under a deprecation heading with a migration
table.

One consequence worth knowing: `experimental/stream/detect-stream-commit.ts` is now a **frozen
duplicate** of `internal/detect-stream-commit.ts`. The maintained version reports its outcome rather
than resolving void (§7.3) and takes the caller's signal rather than an attempt — incompatible
signatures, and the old one is exported from a released subpath. Both files say so; fix bugs in both,
and delete the duplicate along with the subpath.

### Kept unchanged

- `createRetryableModel` and `BaseRetryableModel` — **untouched**, not reused by the new API
- `ai-retry/language-model`, `/embedding-model`, `/image-model` and their `/conditions` subpaths
- `ai-retry/retryables`
- Internals reused by the new loop: `detectStreamCommit` (becomes internal), `evaluateError`,
  `findRetryModel`, `resolveBackoffDelay`, `calculateExponentialBackoff`, `countModelAttempts`,
  `prepareRetryError`, `resolveModel`, `parseReset`, `createRetryTelemetry`

### Reverted from the discarded branch

| file                               | why it was changed                      | why revert                                                  |
| ---------------------------------- | --------------------------------------- | ----------------------------------------------------------- |
| `internal/base-retryable-model.ts` | sticky state keyed by model             | motivated by a shared driver that no longer exists (§9.4)   |
| `types.ts`                         | deprecated `Retry.options.prompt`       | wrong for the model API, where it works correctly (§6.4)    |
| `experimental/call/*`              | per-run model, `telemetry` field, tests | superseded; the released file is kept frozen and deprecated |
| `internal/telemetry.test.ts`       | `operation: 'call'` coverage            | tests a deleted driver                                      |
| `package.json`, `README.md`        | `/experimental/wrap` export + docs      | replaced by the new surface                                 |

---

## 11. Implementation sequence

All steps below are **done**. Kept for the ordering rationale (types first, then the loop, then the
type tests before the runtime tests).

1. ~~Reset branch to `main`, preserving only the `.test-d.ts` probes.~~
2. Add defaulted `INPUT` to `Retry` / `Retryable` / `Retries` / `Condition` /
   `OnRetryOverrides`. Non-breaking; verify the model API's existing tests still pass.
3. Define per-entry-point `INPUT` types: `GenerateTextInput`, `StreamTextInput`, `EmbedInput`,
   `EmbedManyInput`, `GenerateImageInput` — each a `Partial<Pick<Args, …>>`.
4. Build `runRetryLoop` (error-based only first) + the `EntryPoint` table + the two deadline
   strategies.
5. Wire the five public functions and their hand-written / free signatures.
6. **Type tests first** — port the probes to the real functions, asserting against direct calls.
7. Runtime tests: fail-over, deadlines per entry point, caller cancel, backoff, hooks, telemetry,
   `RetryError`, `Retry.options` overrides per field.
8. Result-based retries: `detectStreamCommit` returns an outcome; loop result branch; normalized
   `finishReason` on the attempt; `onSuccess` with a result.
9. Mark `schemaInvalid()` `@deprecated`.
10. Delete `wrapRetryable`; deprecate `createRetryableCall` / `createRetryableStream` (see §10).
11. README: new section for the five functions; remove the experimental call-level section.

---

## 12. Open questions — resolved

1. **Export path.** Root `ai-retry` barrel, alongside the existing `createRetryable*`. They are the
   headline API and the barrel already imports `ai` at runtime, so a subpath bought nothing.
2. **`retry` naming.** Kept `retry`, with the **bare array documented first** and the object form
   presented as the escape hatch for hooks and telemetry. The stutter only appears in the less
   common form.
3. **`result(predicate)` at the call layer.** Supported, best-effort, and documented as such rather
   than deferred. The loop reconstructs a provider-shaped result from what a call-level attempt can
   observe: `finishReason`, `usage`, `warnings` and `providerMetadata` are translated back, and
   `content` carries the generated text, reasoning and tool calls (§13.5). `finishReason(...)` is
   the supported path and reads a normalized field, so it does not depend on the reconstruction at
   all.
4. **Positive-hook naming for streams.** One `onSuccess`, documented as firing at the boundary that
   can still fail over — the completed call for four entry points, the commit point for
   `streamText`. Two names for one concept cost more than the precision bought, and the user's
   standing preference is fewer exports.
5. **`embedMany` + `maxParallelCalls`.** A retry re-runs the **whole call**: one failed sub-call
   re-embeds every value, not just the failed batch. Stated in the `retryableEmbedMany` JSDoc and
   pinned by a test. Per-batch recovery would need the loop to reach inside the entry point, which
   is exactly the coupling this design avoids.
6. **`transcribe` / `generateSpeech`.** Out, for now. Both are one table row and one free signature
   away, and nothing in the design blocks them — but neither has a retry story anyone has asked for,
   and a smaller surface is easier to change.

---

## 13. What changed during implementation

Four decisions were forced by measurement once the code existed. Each is load-bearing.

### 13.1 `INPUT` defaults to the entry point's input type, not `never`

§4.6 called the `never` default "load-bearing". That is true of the **retryable's** own `INPUT` —
`Condition.switch<INPUT = never>` still defaults to `never`, which is what keeps a retryable that
sets no options assignable everywhere. It is **not** true of the entry point's type parameter.

With `INPUT extends GenerateTextInput = never` on the signature, a caller who used only `onRetry`
(never setting `Retry.options`) got `INPUT = never` and could not return any overrides. Defaulting
to the bound instead fixes that and rejects nothing it should accept: an override belonging to a
different entry point still violates the constraint.

The same reasoning applies to `Retry` and `Retryable` written **bare**. `Retryable<LanguageModel>`
is a published type that users annotate standalone retryables with; defaulting its `INPUT` to
`never` silently broke every such annotation that set `options`. Both now default to the
provider-level call options — the meaning they always had — and only inference produces `never`.

### 13.2 `onRetry` is typed against the bound, not the inferred `INPUT`

Caught by a type error in the runtime tests. `onRetry`'s return type is a covariant inference site,
so it competed with the `retries` array to define `INPUT`: a retry setting
`options: { prompt, temperature }` alongside an `onRetry` returning `{ prompt }` made TypeScript
pick `{ prompt }` and reject the retry.

`RetryOptions` therefore carries a fourth parameter, `OVERRIDE` — the entry point's input type,
passed in directly rather than inferred. `retries` uses `INPUT`; `onRetry` uses `OVERRIDE`. Neither
competes, and `onRetry` no longer has to repeat every field some listed retry happens to set.

### 13.3 The provider-shaped result is reconstructed, not just the finish reason

§7.4 specified normalizing `finishReason` onto the attempt, which is done — and `finishReason()` now
reads that field, so it is identical in both APIs. But `RetryResultAttempt.result` is typed
`LanguageModelResult` and something has to go there. Rather than hand over an SDK result under a
provider-shaped type, `toProviderResult` translates what a call-level attempt can observe.

The one place the two shapes genuinely differ is usage: the SDK flattens the breakdowns into
sibling `*Details` objects, the provider nests them under the totals. That is a rename, and
`toProviderUsage` does it.

### 13.5 The reconstructed result carries content (bug, found in review)

> **Superseded by [§14](#14-call-layer-conditions-with-a-discriminated-result-union).** The
> reconstruction is gone — a call-level condition now receives the entry point's own result — and
> `toProviderResult` / `toProviderContent` were deleted with it. Kept here because the bug is the
> clearest argument for why: an approximation of someone else's type is a drift surface, and this is
> what drifting looked like.

The first cut of `toProviderResult` hardcoded `content: []`, with a comment claiming neither caller
had content to map. That is true of `streamText` — judged before its first content part, it has
produced none by definition — and **false** of `generateText`, where a completed generation has all
of it. The effect was silent: `result((res) => res.content…)` under `retryableGenerateText` saw an
empty array for a perfectly good generation.

`toProviderContent` now translates the parts that have a faithful provider counterpart:

| part            | translation                                                                                                                                                         |
| --------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `text`          | passes through — the two shapes are identical                                                                                                                       |
| `reasoning`     | passes through — likewise                                                                                                                                           |
| `tool-call`     | `input` is re-serialized; the provider carries the raw JSON string the model emitted, and the SDK has already parsed it                                             |
| everything else | dropped — files, sources, tool results, approvals and custom parts differ in more than naming, and an approximation that looks real is worse than a visible absence |

`streamText` still passes no content, but now because it genuinely has none rather than by
assumption, and the call site says so.

Two regression tests pin it (`should give a result condition the generated content`, `… the tool
calls that were made`), both verified to fail against the pre-fix code. A third pins that a stream
judged pre-commit still sees an empty `content`, so the two cases cannot be conflated again.

---

### 13.4 `disabled` means "call the entry point directly"

Including the SDK's own `maxRetries` default. The alternative — keeping the loop's `maxRetries: 0`
while skipping the retries — would make `disabled: true` behave like neither the retry API nor a
plain call.

---

## 14. Call-layer conditions with a discriminated result union

**Status: implemented.** This replaced the reconstruction described in §13.5, which is gone —
`to-provider-result.ts` was deleted along with the bug class it carried. §14.6 records the two probe
findings the implementation then contradicted; §14.8 records what changed.

### 14.1 The problem it solves

Result conditions were written against the _provider_ result, because that is where retries
originally ran. At the call layer there is no provider result, so §13.5 reconstructed one — text,
reasoning and tool calls translated, files/sources/tool-results dropped, usage renamed. That
reconstruction was:

- **lossy** — a `result()` predicate cannot see files, sources or tool results at all;
- **a drift surface** — it must track two independently evolving type sets, and already produced one
  silent bug (`content: []` for a completed `generateText`);
- **awkward where it is most needed** — `embed` reads `res.embeddings[0]`, `generateImage` reads
  `res.images[0]`, because the provider shape is plural and batch-oriented.

### 14.2 The design

A **call-layer-only** condition API, bound per family, whose result is a discriminated union over
the _operation_. The model-layer conditions stay untouched and are deprecated separately.

```ts
type CallLanguageModelResult<TOOLS extends ToolSet = ToolSet> =
  | GenerateTextResultInfo<TOOLS> // operation: 'generateText'
  | StreamTextResultInfo; // operation: 'streamText'

type CallEmbeddingModelResult = EmbedResultInfo | EmbedManyResultInfo;
type CallImageModelResult = GenerateImageResultInfo; // one member, no guard needed
```

Fields common to every member of a family are reachable **without narrowing** (`finishReason`,
`usage` for language). Member-specific fields need a discriminant check or a guard:

```ts
result((res) => {
  if (res.finishReason === 'content-filter') return true; // no guard
  if (isGenerateTextResult(res)) return res.text.length < 10; // narrowed
  return false;
});
```

Three properties make this better than both alternatives considered before it (§14.5):

1. **No new generic.** The result type is a function of `MODEL`, which `RetryContext` already
   carries, so `Condition` / `Retryable` / `RetryContext` need no third parameter. This is what
   §7.4 balked at, and the union avoids it.
2. **`streamText` stops being a hole.** Its member simply declares no content, because none exists
   before the commit point. The streaming limitation becomes a fact in the type rather than a
   caveat in prose. (Binding to the SDK's own `StreamTextResult` is impossible: every field is a
   `PromiseLike` that settles only once the stream is consumed — precisely what commit detection
   must not do.)
3. **No faking.** `toProviderResult` / `toProviderContent` disappear, and with them the bug class
   of §13.5.

### 14.3 Keeping the two layers apart

**This is the part that decides the design.** If the call-layer result were reached through the
existing `RetryContext<MODEL>`, both layers would produce the _identical_ context type, and a
call-layer condition would silently typecheck against a model-layer `retries` array and vice versa.

So the call layer needs its own context type, not merely its own exports:

| layer                             | context                           | result          |
| --------------------------------- | --------------------------------- | --------------- |
| model (`createRetryableModel`)    | `RetryContext<MODEL>` — unchanged | provider-shaped |
| call (`retryableGenerateText`, …) | `CallRetryContext<MODEL>` — new   | the union       |

Measured, and it holds structurally rather than by convention: the call-layer attempt carries the
entry point's arguments (`prompt: string | Array<ModelMessage>`) where the model-layer one carries
provider call options (`prompt: LanguageModelV4Prompt`), so neither context is assignable to the
other and the contravariant parameter position of a retryable rejects each layer's conditions from
the other's list, in both directions.

### 14.4 What is shared, what is new

`Condition` and the combinators carry a **layer tag defaulted to `'model'`**, so `Condition<MODEL>`
keeps working and the model-layer conditions compile untouched.

The tag is a string literal (`'model' | 'call'`) rather than the context type itself, because the
context is a function of `MODEL` and `MODEL` is only fixed at the individual condition, not at the
factory that builds it. `LayerContext<LAYER, MODEL>` and `LayerRetryable<MODEL, INPUT, LAYER>` map
the tag to the real types — the standard defunctionalized stand-in for the higher-kinded parameter TS
does not have.

| piece                                                                                               | treatment                                                                                                                         |
| --------------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------- |
| `Condition`, `and` / `or` / `not`                                                                   | generic over `LAYER`, default `'model'`                                                                                           |
| error API (`error`, `httpStatus`, `timeout`, `aborted`), `noImage`                                  | one implementation, instantiated per layer by the factory — their logic reads only `ctx.current.error`, nothing layer-specific    |
| result API + guards                                                                                 | new, bound to the union                                                                                                           |
| `Retryable` / `Retries`                                                                             | call-layer versions (`CallRetryable` / `CallRetries`); the model ones stay frozen. `Retry` is context-free and is shared as-is    |
| `findRetryModel`, `evaluateError`, `resolveBackoffDelay`, `prepareRetryError`, `countModelAttempts` | driven by one structural `RetryContextLike`; `retries` threaded as the union `RetriesLike = Retries \| CallRetries` — see §14.7.2 |

Layout, mirroring the existing one:

```
src/call/
  types.ts                             CallRetryContext, CallRetryable, CallRetries, CallArgs, the result unions
  guards.ts                            the is*Result narrowing
  tag-result.ts                        the Proxy that presents a result as its union member
  conditions/result.ts                 the call-layer result API
  language-model/conditions/index.ts   → ai-retry/call/language-model/conditions
  embedding-model/conditions/index.ts  → ai-retry/call/embedding-model/conditions
  image-model/conditions/index.ts      → ai-retry/call/image-model/conditions
  <family>-model/functions/…           the entry points, beside their family's conditions
```

The `types.ts` / `guards.ts` split mirrors the root (`src/types.ts` + `src/internal/guards.ts`);
`tagResult` has its own module so it can be tested in isolation, which matters more than usual —
its whole reason for existing is a failure mode types cannot catch (§14.8.3).

`Condition`, the combinators and the error API stay in `src/internal/conditions/` and serve both
layers; only the result API is new code. `src/internal/detect-stream-commit.ts` moved to
`src/call/`, since it only ever served the call layer and now reports a call-layer type.

The subpaths are verbose while both APIs exist; the short names free up when the model layer goes.

### 14.5 Alternatives rejected

| option                                            | why not                                                                                                                                                                                                                                          |
| ------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| Keep provider-shaped results (status quo, §13.5)  | lossy, permanent drift surface, plural/batch shapes are wrong for `embed` and `generateImage`                                                                                                                                                    |
| Bind `result()` to the SDK result per entry point | 7 exports instead of 3; **impossible for `streamText`** (all-`PromiseLike` result). It would have typed tool calls without the caller naming a tool set, which the union cannot — the union's `TOOLS` has to be named at the condition (§14.8.1) |
| Infer the result type from the `retries` array    | **measured impossible** — `result(pred)` fixes its parameter at its own call, and TS will not push a contextual type backwards across the intervening `.switch()`. `res` comes out `unknown`. Same wall as `MODEL` (§4.6)                        |
| A builder form, `retry: (c) => [c.result(…)]`     | measured to work and the only mechanism that gives full inference, but it adds a second `retry` form. Additive later if wanted — `retry` widening to accept a callback is not breaking                                                           |

### 14.6 Measured findings

Two throwaway probes preceded the implementation. Both passed; **two of their claims did not survive
contact with the real types**, and both failures were the probe encoding a wrong expectation rather
than the design being wrong. The live claims are now carried by
`src/call/conditions/conditions.test-d.ts` (19 type tests).

**Probe A — the union through a conditional type.** The risk was that `CallResult<MODEL>` stays a
deferred conditional and refuses to narrow. It does not:

- the conditional resolves to the family union once the bound export pins `MODEL` ✓
- common fields readable with no narrowing ✓
- narrowing works on the discriminant _and_ through a guard function ✓
- a member-specific field without narrowing is rejected ✓
- `embed` and `embedMany` narrow to their own shapes from one export ✓
- a single-member family (image) needs no guard at all ✓
- ~~`isGenerateTextResult<TOOLS>(res)` types `toolCalls[].toolName` to the actual tool names~~ —
  **wrong, see §14.8.1.** A type argument on the _guard_ is silently ignored; the tool set has to be
  named where the result type enters, at `result<TOOLS>(…)`.

**Probe B — the context split.**

- `Condition<MODEL>` still means what it did before the layer parameter ✓
- model-layer conditions compile unchanged ✓
- `and`/`or`/`not` follow either layer without duplication ✓
- a model-layer condition is **rejected** by a call-layer `retry` ✓
- a call-layer condition is **rejected** by a model-layer `retries` ✓
- ~~a combinator handed conditions from both layers is rejected~~ — **wrong, see §14.8.2.** `and()`
  accepts the mix; the mistake surfaces one step later, at the `retries` list.

The recurring trap is worth naming, because it caught two probes in a row: a probe that stands in for
the real type can encode the answer you expect. The first run of Probe A appeared to show `TOOLS`
typing failing because `type MyTools = {weather: …} & ToolSet` widens `keyof` back to `string` — a
wrong _failure_. The finding that replaced it was a wrong _success_ for the same reason. §4.3's rule
— check against the real value, not a stand-in — applies to probes, not just to assertions.

### 14.7 Open questions — resolved

1. **Field sets.** Resolved by not choosing: each member is `{ operation } & <the SDK's own result>`,
   so the library owns no field list at all and nothing can drift. The one exception is
   `StreamTextResultInfo`, which cannot be the SDK's `StreamTextResult` (every field is a promise
   that settles only on consumption) and so declares the three values read off the stream's terminal
   parts: `finishReason`, `usage`, `providerMetadata`.
2. **Generalizing the shared internals** turned out to be the smallest part, not the largest. They
   read only `current.type`, `current.model` and each attempt's `model`, so a single structural
   `RetryContextLike` covers both layers, and `Retries`/`CallRetries` are threaded as a union
   (`RetriesLike`) rather than a loosened shape — a union keeps alias-based inference working, which
   a loosened shape broke.
3. **Subpath naming.** Kept as designed: `ai-retry/call/<family>-model/conditions`.
4. **Model-layer result support for embedding/image.** Still missing, and now the sharper gap:
   `result()` works for every family at the _call_ level, but `retryable-embedding-model.ts` and
   `retryable-image-model.ts` still have no result branch, so a result condition for those families
   silently never fires under `createRetryableModel`.

### 14.8 What changed during implementation

**14.8.1 A type argument on a guard is ignored; the tool set is named at the condition.**
`isGenerateTextResult<TOOLS>(res)` does nothing: narrowing a value filters the _declared_ union
against the predicate type, and the declared union wins. `TOOLS` has to be pinned where the result
type is introduced — `result<typeof tools>((res) => …)` — which is why the language `result` leads
with a `TOOLS` parameter that the generic one does not have.

Two further corrections came out of measuring this properly, both by checking a direct
`generateText` call as the baseline:

- `TypedToolCall<TOOLS> = StaticToolCall<TOOLS> | DynamicToolCall`, and `DynamicToolCall` has
  `toolName: string`, so **`toolCalls[].toolName` is `string` on a direct call too** until `dynamic`
  is discriminated. This layer is not worse than the SDK; the earlier claim that typed tool calls
  were "the one thing no other option delivers" overstated what the SDK itself delivers.
- Once `dynamic` is discriminated, naming `TOOLS` does pay off: `toolName` narrows to the literal and
  `input` to the schema type. So the parameter earns its place — just not on the guard.

**14.8.2 Mixed-layer combinators are caught later, not at the combinator.** `and(callCondition,
modelCondition)` compiles; what it produces belongs to neither layer and is rejected by whichever
`retries` list it is written into. Rejecting it at `and()` would need the layer parameter to be
invariant, which is not worth the cost for a mistake that is still caught.

**14.8.3 The result is tagged with a `Proxy`, not a spread.** `{ operation, ...result }` looked
obvious and is wrong: the SDK's results expose most of themselves through prototype getters, so the
spread yields an object whose `text` and `toolCalls` are silently `undefined`. Caught by a runtime
test, not by types — the spread's type is correct, only its value is empty. `tagResult` forwards
every read to the real result and adds the tag, leaving the caller's own object untouched.

**14.8.4 `embed`/`embedMany` are not generic over the value.** In `ai@7.0.35` both take
`value: string` / `values: Array<string>`, so `EmbedResultInfo` carries no `VALUE` parameter.

**14.8.5 The attempt carries no normalized `finishReason` at the call layer.** It exists on the
model-layer attempt to paper over the provider's nested shape; the call layer's results are already
flat, so `finishReason` is read off the result where the operation has one — and embeddings and
images simply have none. `prepareRetryError` looks in both places.

**14.8.6 `AnyModel` is now one name in one place** (review follow-up). The union
`LanguageModel | EmbeddingModel | ImageModel` was written inline ~40 times across 17 files and
declared locally twice, while `src/internal/conditions/condition.ts` exported a _different_ type
under the same name (`AnyModel = AnyResolvableModel`). It now lives in `src/types.ts` beside
`AnyResolvableModel`, and the conditions name `AnyResolvableModel` directly, so one name means one
thing. Watch the substring hazard when sweeping this: `ResolvableLanguageModel | EmbeddingModel |
ImageModel` contains the union as a suffix, and a blind replace silently produces
`ResolvableAnyModel`.

**14.8.7 Model-layer types carry a `Model` prefix** (review follow-up). Once `CallRetryAttempt`
existed, the unprefixed `RetryAttempt` stopped saying which layer it belonged to. Thirteen types that
are clearly one layer's are now `Model`-prefixed, with the old names kept as deprecated aliases;
`src/types.test-d.ts` pins each alias to be the _same_ type as its replacement so the two cannot
drift while both exist. `Retry`, `OnRetryOverrides`, `Reset` and `RetryTelemetrySettings` keep their
names, being genuinely shared.

The one that mattered independently of symmetry is `CallOptions`, which meant _provider_ call
options — precisely backwards once a call layer existed, and sitting on the model attempt's `options`
field while the call attempt's equivalent was `CallArgs`. It is now `ModelCallOptions`.
`CallArgs` keeps its name rather than claiming the freed `CallOptions`: the deprecated alias
occupies that name for the whole deprecation window, and "arguments" is the more accurate word for an
entry point's whole argument object anyway.

`FailureContext`'s bound was also fixed here — it was
`ResolvableLanguageModel | EmbeddingModel | ImageModel` where both its siblings used
`AnyResolvableModel`, so `FailureContext<ResolvableEmbeddingModel>` was rejected while
`SuccessContext<ResolvableEmbeddingModel>` compiled. Pre-existing on `main`; widening is not
breaking.

**14.8.8 Call-layer names carry the `Call` prefix, and `Args` is the one spelling**
(review follow-up). `RetryOptions` was declared twice with different meanings — the argument shape of
`Condition.retry()` in `internal/conditions/condition.ts`, and the whole call-level retry config in
`call/retry-arg.ts` — and that same file exported `CallSuccessAttempt` _with_ the prefix beside
`RetryArg` and `RetryOptions` without it. Both are now `CallRetryArg` / `CallRetryOptions` (a straight
rename, since neither had shipped).

`Args` and `Arguments` were both in use, including for two exported names that differed only by the
abbreviation: the loop's minimal shape and the per-family entry-point arguments. Now `RetryLoopArgs`
for the first and `CallArgs` for the second, with `Args` everywhere.

**14.8.9 The overridable-argument list stays closed.** Deriving it from the entry point's arguments
was considered and rejected: it would make every argument the SDK adds overridable by default, which
today already admits `output`, `toolChoice`, `stopWhen`, `toolOrder`, `toolsContext` — each of which
changes the result or the tool loop the caller was promised — plus a `telemetry` that collides with
this library's own. The hand-written list is not unchecked: it is only ever used through `Pick`, so
every key must exist on _both_ language entry points, and a misspelling or an SDK rename fails to
compile.

**14.8.10 Type tests assert exact types, not assignability** (review follow-up). `toMatchTypeOf` is
both deprecated and too weak: it checks assignability, so it passes against `any`. Measured — a
mutation making `Condition.switch` return `any` failed 6 assertions under `toMatchTypeOf` and 24
under `toEqualTypeOf`. Switching also surfaced something the weak assertion had been hiding:
`ModelRetryable<M>` defaults `INPUT` to the provider-level overrides while `CallRetryable<M>`
defaults it to `never`, so the exact return of an unbound `.switch()` is `ModelRetryable<M, never>`.

Coverage is now wired (`pnpm test:coverage`, v8, gated). It found three genuine gaps, all since
closed: the `?? providerMetadata` fallback in commit detection, the backoff delay on the _result_
retry path (only the error path was covered), and the error-attempt guard in the _generic_ result
factory (only the language factory's copy was). `src/call` is at 99.4% statements / 100% lines; what
remains is one defensive `.catch()` on a stream cancel that cannot be provoked.

**14.8.11 Result-based retries were extended to embed, embedMany and generateImage.** They previously
had no `settle`, so a returned result was always terminal. All five entry points now report one, which
is what makes `result()` meaningful for those families — a degenerate embedding or too few images is
not an error and nothing else could catch it.

---

## Evidence index

The six standalone probes have been **replaced** by a `.test-d.ts` beside each entry point, which
makes the same claims against the real functions instead of `declare function` stand-ins. Each
claim below names the test that now carries it; all run under `pnpm test`
(vitest `typecheck.enabled: true`).

| original probe                  | proved                                                                                                                  | now carried by                                                                                     |
| ------------------------------- | ----------------------------------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------- |
| `wrap-retryable.test-d.ts`      | identity signature preserves inference; `Parameters`-derived args do not                                                | superseded — the hand-written signature (§4.3) is what shipped                                     |
| `dedicated-signature.test-d.ts` | hand-written signature + `ReturnType<typeof streamText<TOOLS>>` matches a direct call while returning a promise         | "should keep the result type identical to a direct call", "should return a promise"                |
| `merged-args.test-d.ts`         | merging `retry` preserves inference for both language entry points; excess property checking survives; bare array works | "should reject an unknown argument", "should accept the bare-array shorthand"                      |
| `non-generic-args.test-d.ts`    | `Parameters<typeof embed>[0] & retry` is exact and free; wrong-family fallbacks rejected                                | `embed` / `embed-many` / `generate-image` `.test-d.ts`                                             |
| `input-generic.test-d.ts`       | `INPUT` is not the `MODEL` trap; a no-option retryable stays portable; foreign overrides rejected                       | "should accept/reject overrides…" per entry point; "should share a retryable that sets no options" |
| `combined-generics.test-d.ts`   | `TOOLS` and `INPUT` coexist in one args object                                                                          | "should narrow activeTools against the tools map" with a typed `retry` present                     |

One claim is new, and was not measurable before the implementation existed: **threading `RESULT`
into the args object does not disturb `TOOLS` inference** — `onSuccess` receives the same result
type a direct call produces ("should type onSuccess with the entry point result"). That is what
made §7.6's "`onSuccess` with a result" affordable.

Runtime findings that motivated the design, still worth re-checking if `generateObject` /
`streamObject` are ever brought into scope:

- `generateObject` silently ignores an injected `timeout` (1018 ms vs `generateText`'s 113 ms), so
  a generic wrapper cannot apply deadlines uniformly (§6.1)
- `streamObject.partialObjectStream` yields 0 partials after commit detection
  (`ERR_INVALID_STATE`), because its `fullStream` is not a fresh tee (§8)
