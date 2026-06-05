import { streamText } from 'ai';
import type { ToolSet } from 'ai';
import {
  createRetryableStream,
  type RetryableStreamOptions,
} from '../stream/create-retryable-stream.js';

/**
 * `streamText`'s arguments and result, parametrized by the tool set so tool
 * typing flows through to the caller. The instantiation expression
 * (`streamText<TOOLS>`) leaves `OUTPUT` at its default; structured-output
 * (`output`/`experimental_output`) typing is not threaded through.
 */
type StreamTextArgs<TOOLS extends ToolSet> = Parameters<
  typeof streamText<TOOLS>
>[0];
type StreamTextReturn<TOOLS extends ToolSet> = ReturnType<
  typeof streamText<TOOLS>
>;

/**
 * Options for {@link createRetryableStreamText}. The base model, the retry
 * handlers, and the lifecycle hooks (a `classifyPart` override is accepted but
 * rarely needed — the default handles the `streamText` `fullStream` protocol).
 *
 * Note: a retry's `options.prompt` (and an `onRetry` prompt override) is
 * ignored here — see {@link createRetryableStreamText}. The prompt is set on the
 * `streamText` call arguments instead.
 */
export type RetryableStreamTextOptions = RetryableStreamOptions;

/**
 * A drop-in for `streamText` that retries the whole call. Takes the same
 * arguments as `streamText` (minus `model`, which the retryable supplies) and
 * resolves once an attempt commits — i.e. produces its first content — so the
 * winning attempt's result is the one returned.
 *
 * Generic over the tool set, inferred per call from `tools`, so the resolved
 * `StreamTextResult` carries the caller's tool types.
 */
export type RetryableStreamText = <TOOLS extends ToolSet = ToolSet>(
  args: Omit<StreamTextArgs<TOOLS>, 'model'>,
) => Promise<StreamTextReturn<TOOLS>>;

/**
 * Make a `streamText` call retryable at the call level — a typed `streamText`
 * drop-in built on {@link createRetryableStream}.
 *
 * Unlike wrapping a model with `createRetryable` (which retries below
 * `streamText`, at `doStream`), this re-runs the entire `streamText` call with
 * the next model when an attempt fails *before* producing content. That is the
 * only place a `streamText`-level deadline — `timeout.chunkMs`/`stepMs`/
 * `totalMs` or an inbound `abortSignal` — can fail over: once the call's master
 * signal aborts, `streamText` finalizes its stream as aborted and discards
 * anything a lower retry produces underneath (see issue #50). Re-running gives
 * the next attempt a fresh master signal.
 *
 * Scope: errors and deadline aborts that occur before the first content part.
 * Result-based switches (content-filter, schema) still recover best below
 * `streamText` — pass a `createRetryable(...)` as the `model` for those.
 *
 * Prompt overrides do not apply here. A retry's `options.prompt` (and an
 * `onRetry` prompt override) is the low-level `LanguageModelV3Prompt` format,
 * which cannot be mapped onto `streamText`'s high-level `prompt`/`messages`, so
 * it is ignored. Set the prompt on the call arguments instead. Every other
 * override (temperature, headers, providerOptions, …) is forwarded normally.
 *
 * Trade-offs:
 * - The call is `await`ed: it resolves once content commits, not synchronously
 *   like `streamText`.
 * - Commit is detected by reading a teed branch of the result's `fullStream`
 *   up to the first content part; the caller drives the body (via `fullStream`,
 *   `toUIMessageStreamResponse()`, etc.), so back-pressure past the commit
 *   point is preserved and only the leading parts are pre-read.
 * - Your `streamText` callbacks pass through and fire per attempt — including
 *   one that is later recovered. Use the retry-level `onError`/`onRetry` hooks
 *   to observe only the final, committed outcome. `onError` defaults to a
 *   no-op (rather than `streamText`'s `console.error`) so recovered attempts
 *   are not logged.
 * - After the first content part, an attempt is committed and cannot fail over
 *   (re-running would duplicate output) — the error/abort flows through as it
 *   would for a plain `streamText` call.
 */
export function createRetryableStreamText(
  options: RetryableStreamTextOptions,
): RetryableStreamText {
  const retryableStream = createRetryableStream(options);

  return <TOOLS extends ToolSet = ToolSet>(
    args: Omit<StreamTextArgs<TOOLS>, 'model'>,
  ) =>
    retryableStream<StreamTextReturn<TOOLS>>(
      (attempt) => {
        /**
         * Strip `prompt`: the driver carries it in the low-level
         * `LanguageModelV3Prompt` format (valid when the retry wraps
         * `doStream`/`doGenerate` directly), which would clobber `streamText`'s
         * high-level `prompt`/`messages` from `args`. The remaining overrides
         * (temperature, headers, providerOptions, …) are valid call settings
         * and pass through unchanged.
         */
        const { prompt: _prompt, ...callOverrides } = attempt.options;

        return streamText({
          ...args,
          ...callOverrides,
          model: attempt.model,
          abortSignal: attempt.abortSignal,
          /**
           * Default `onError` to a no-op. This wrapper manages errors itself
           * (it detects them from `fullStream`, reports them through the
           * retry-level `onError`/`onRetry` hooks, and surfaces the final one
           * via the rejected promise or the returned stream), so `streamText`'s
           * default `console.error` logger would just spam every recovered
           * attempt. A caller-provided `onError` is respected and fires per
           * attempt — including one that is later recovered.
           */
          onError: args.onError ?? (() => {}),
        } as StreamTextArgs<TOOLS>);
      },
      { abortSignal: args.abortSignal },
    );
}
