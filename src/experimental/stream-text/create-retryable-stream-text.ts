import { streamText } from 'ai';
import type { ToolSet } from 'ai';
import {
  createRetryableCall,
  type RetryableCallOptions,
} from '../call/create-retryable-call.js';

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
 * Options for {@link createRetryableStreamText}. Identical to the generic
 * driver's options: a base model, the retry handlers, and the lifecycle hooks.
 *
 * Note: a retry's `options.prompt` (and an `onRetry` prompt override) is
 * ignored here — see {@link createRetryableStreamText}. The prompt is set on the
 * `streamText` call arguments instead.
 */
export type RetryableStreamTextOptions = RetryableCallOptions;

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
 * Make a `streamText` call retryable at the call level.
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
 * - To detect commit, the adapter pumps a teed branch of the stream only up to
 *   the first content part, then stops; the caller drives the body (via
 *   `fullStream`, `toUIMessageStreamResponse()`, etc.), so back-pressure past
 *   the commit point is preserved and only the leading parts are pre-read.
 * - After the first content part, an attempt is committed and cannot fail over
 *   (re-running would duplicate output) — the error/abort flows through as it
 *   would for a plain `streamText` call.
 */
export function createRetryableStreamText(
  options: RetryableStreamTextOptions,
): RetryableStreamText {
  const run = createRetryableCall(options);

  return <TOOLS extends ToolSet = ToolSet>(
    args: Omit<StreamTextArgs<TOOLS>, 'model'>,
  ) =>
    run<StreamTextReturn<TOOLS>>(
      (attempt) =>
        new Promise<StreamTextReturn<TOOLS>>((resolve, reject) => {
          /** Set once the attempt's outcome is decided (committed or failed). */
          let settled = false;

          /**
           * Strip `prompt`: the driver carries it in the low-level
           * `LanguageModelV3Prompt` format (valid when the retry wraps
           * `doStream`/`doGenerate` directly), which would clobber
           * `streamText`'s high-level `prompt`/`messages` from `args`. The
           * remaining overrides (temperature, headers, providerOptions, …) are
           * valid call settings and pass through unchanged.
           */
          const { prompt: _prompt, ...callOverrides } = attempt.options;

          const result = streamText({
            ...args,
            ...callOverrides,
            model: attempt.model,
            abortSignal: attempt.abortSignal,
            onChunk(event) {
              if (!settled) {
                settled = true;
                resolve(result);
              }
              return args.onChunk?.(event);
            },
            onError(event) {
              /**
               * A pre-content error fails the attempt so the driver can fail
               * over; the driver reports it through the retry `onError` hook.
               * Once committed, forward it to the caller's handler instead.
               */
              if (!settled) {
                settled = true;
                reject(event.error);
                return;
              }
              return args.onError?.(event);
            },
            onAbort(event) {
              if (!settled) {
                settled = true;
                reject(attempt.abortSignal?.reason ?? new Error('aborted'));
                return;
              }
              return args.onAbort?.(event);
            },
            onFinish(event) {
              /**
               * A finish before any content (e.g. an empty completion) is a
               * successful commit, not a failure.
               */
              if (!settled) {
                settled = true;
                resolve(result);
              }
              return args.onFinish?.(event);
            },
          } as StreamTextArgs<TOOLS>);

          /**
           * Pump a teed branch only until the outcome is decided, then cancel
           * it. The leading parts are read here to fire the callbacks above;
           * the body is left for the caller to drive. Cancelling this branch
           * does not cancel the source — the caller's own consumption of the
           * returned result still receives every part, including the ones read
           * here (the tee buffers them).
           */
          void (async () => {
            const reader = result.fullStream.getReader();
            try {
              while (!settled) {
                const { done } = await reader.read();
                if (done) break;
              }
            } catch {
              /* the callbacks own the outcome; ignore read errors here */
            } finally {
              void reader.cancel().catch(() => {});
            }
          })();
        }),
      { abortSignal: args.abortSignal },
    );
}
