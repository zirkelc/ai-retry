import { BaseRetryableModel } from './base-retryable-model.js';
import { evaluateError } from './evaluate-error.js';
import { findRetryModel } from './find-retry-model.js';
import { resolveLanguageModel } from './resolve-model.js';
import { mergeLanguageModelCallOptions } from './merge-retry-call-options.js';
import { createRetryTelemetry, type RetryTelemetry } from './telemetry.js';
import { resolveBackoffDelay } from './resolve-backoff-delay.js';
import {
  isCallAbort,
  isRetryCancelled,
  waitBeforeRetry,
} from './retry-signal.js';
import { totalTimeoutMs } from './retry-timeout.js';
import type {
  LanguageModel,
  LanguageModelCallOptions,
  LanguageModelResult,
  LanguageModelStream,
  LanguageModelStreamPart,
  OnRetryOverrides,
  Retry,
  ModelRetryAttempt,
  ModelRetryContext,
  ModelRetryResultAttempt,
} from '../types.js';
import {
  isErrorAttempt,
  isGenerateResult,
  isStreamContentPart,
} from './guards.js';

export class RetryableLanguageModel
  extends BaseRetryableModel<LanguageModel>
  implements LanguageModel
{
  readonly specificationVersion = 'v4';

  get modelId() {
    return this.currentModel.modelId;
  }

  get provider() {
    return this.currentModel.provider;
  }

  get supportedUrls() {
    return this.currentModel.supportedUrls;
  }

  /**
   * Execute a function with retry logic for handling errors
   */
  private async withRetry<
    RESULT extends LanguageModelStream | LanguageModelResult,
  >(input: {
    fn: (retryCallOptions: LanguageModelCallOptions) => Promise<RESULT>;
    callOptions: LanguageModelCallOptions;
    attempts?: Array<ModelRetryAttempt<LanguageModel>>;
    currentRetry?: Retry<LanguageModel>;
    recorder?: RetryTelemetry;
  }): Promise<{
    result: RESULT;
    attempts: Array<ModelRetryAttempt<LanguageModel>>;
    callOptions: LanguageModelCallOptions;
    /**
     * For stream results: the still-open attempt span number, to be closed by
     * the caller once the consumption outcome is known. Undefined otherwise.
     */
    pendingAttempt?: number;
    /**
     * For stream results: the retry the returned stream runs under, if it
     * came from a failover, so the caller judges the stream against it.
     */
    currentRetry?: Retry<LanguageModel>;
  }> {
    /**
     * Track all attempts.
     */
    const attempts: Array<ModelRetryAttempt<LanguageModel>> =
      input.attempts ?? [];

    /**
     * Track current retry configuration.
     */
    let currentRetry: Retry<LanguageModel> | undefined = input.currentRetry;

    while (true) {
      /**
       * The previous attempt that triggered a retry, or undefined if this is the first attempt
       */
      const previousAttempt = attempts.at(-1);

      /**
       * Call the onRetry handler if provided.
       * Skip on the first attempt since no previous attempt exists yet.
       */
      let onRetryOverrides: OnRetryOverrides<LanguageModel> | undefined;
      if (previousAttempt) {
        const currentAttempt: ModelRetryAttempt<LanguageModel> = {
          ...previousAttempt,
          model: this.currentModel,
        };

        /**
         * Create a shallow copy of the attempts for testing purposes
         */
        const updatedAttempts = [...attempts];

        const context: ModelRetryContext<LanguageModel> = {
          current: currentAttempt,
          attempts: updatedAttempts,
        };

        onRetryOverrides = (await this.options.onRetry?.(context)) ?? undefined;
      }

      /**
       * Get the retry call options overrides for this attempt
       */
      const retryCallOptions = mergeLanguageModelCallOptions({
        callOptions: input.callOptions,
        currentRetry,
        onRetryOverrides,
      });

      /**
       * The model and 1-based index for this attempt, captured for telemetry
       * before the call is issued.
       */
      const attemptModel = this.currentModel;
      const attemptNumber = attempts.length + 1;
      input.recorder?.startAttempt({
        attempt: attemptNumber,
        provider: attemptModel.provider,
        modelId: attemptModel.modelId,
        timeoutMs: totalTimeoutMs(currentRetry?.timeout),
      });

      try {
        /**
         * Call the function that may need to be retried, with the attempt as
         * the ambient span so the provider's own spans nest inside it rather
         * than beside the retry tree.
         */
        const result = await (input.recorder
          ? input.recorder.withAttempt(attemptNumber, () =>
              input.fn(retryCallOptions),
            )
          : input.fn(retryCallOptions));

        /**
         * Check if the result should trigger a retry (only for generate results, not streams)
         */
        if (isGenerateResult(result)) {
          const { retryModel, attempt } = await this.handleResult(
            result,
            attempts,
            retryCallOptions,
          );

          /**
           * If the inbound abort signal is already aborted and the chosen
           * retry does not supply a fresh deadline, the retry would die
           * instantly. Unlike the error path there is no error to rethrow:
           * the result stands, as a finish part does on the stream path.
           */
          if (
            retryModel &&
            !isRetryCancelled(input.callOptions.abortSignal, retryModel)
          ) {
            /**
             * Only record the attempt once it is known to be retried. A
             * result that ends the loop is the successful outcome, not a
             * failed attempt, so it stays out of the attempts list.
             */
            attempts.push(attempt);

            const calculatedDelay = resolveBackoffDelay(retryModel, attempts);

            input.recorder?.endAttempt({
              attempt: attemptNumber,
              outcome: 'retry',
              finishReason: result.finishReason.unified,
              delayMs: calculatedDelay,
            });

            await waitBeforeRetry(
              calculatedDelay,
              input.callOptions.abortSignal,
              retryModel,
            );

            this.currentModel = retryModel.model;
            currentRetry = retryModel;

            /**
             * Continue to the next iteration to retry
             */
            continue;
          }

          input.recorder?.endAttempt({
            attempt: attemptNumber,
            outcome: 'success',
            finishReason: result.finishReason.unified,
          });
          return { result, attempts, callOptions: retryCallOptions };
        }

        /**
         * Stream results are not terminal here: the outcome depends on
         * consumption (the stream may still error or hit a retryable finish
         * before content flows). Leave the attempt span open and hand its
         * number back so the stream wrapper can close it once known.
         */
        return {
          result,
          attempts,
          callOptions: retryCallOptions,
          pendingAttempt: attemptNumber,
          currentRetry,
        };
      } catch (error) {
        const { retryModel, attempt, finalError } = await this.handleError(
          error,
          attempts,
          retryCallOptions,
          input.callOptions.abortSignal,
          currentRetry,
        );

        attempts.push(attempt);

        if (!retryModel) {
          input.recorder?.endAttempt({
            attempt: attemptNumber,
            outcome: 'failure',
            error,
          });
          throw finalError;
        }

        /**
         * If the inbound abort signal is already aborted and the chosen
         * retry does not supply a fresh deadline, the retry would die
         * instantly with the same abort. Rethrow rather than fire a
         * misleading retry against a dead signal.
         */
        if (isRetryCancelled(input.callOptions.abortSignal, retryModel)) {
          input.recorder?.endAttempt({
            attempt: attemptNumber,
            outcome: 'failure',
            error,
          });
          throw error;
        }

        const calculatedDelay = resolveBackoffDelay(retryModel, attempts);

        input.recorder?.endAttempt({
          attempt: attemptNumber,
          outcome: 'retry',
          error,
          delayMs: calculatedDelay,
        });

        await waitBeforeRetry(
          calculatedDelay,
          input.callOptions.abortSignal,
          retryModel,
        );

        this.currentModel = retryModel.model;
        currentRetry = retryModel;
      }
    }
  }

  /**
   * Handle a successful result and determine if a retry is needed
   */
  private async handleResult(
    result: LanguageModelResult,
    attempts: ReadonlyArray<ModelRetryAttempt<LanguageModel>>,
    callOptions: LanguageModelCallOptions,
  ) {
    const resultAttempt: ModelRetryResultAttempt = {
      type: 'result',
      result: result,
      finishReason: result.finishReason.unified,
      model: this.currentModel,
      options: callOptions,
    };

    /**
     * Save the current attempt
     */
    const updatedAttempts = [...attempts, resultAttempt];

    const context: ModelRetryContext<LanguageModel> = {
      current: resultAttempt,
      attempts: updatedAttempts,
    };

    const retryModel = await findRetryModel(
      this.options.retries,
      context,
      resolveLanguageModel,
    );

    return { retryModel, attempt: resultAttempt };
  }

  /**
   * Handle an error and determine if a retry is needed.
   *
   * Returns a `finalError` (and undefined `retryModel`) when no retry
   * matched, so callers can decide how to surface it: throwing for the
   * generate path, or enqueuing a `{ type: 'error' }` stream part for
   * the stream path. If multiple attempts were made, the original error
   * is wrapped in a `RetryError`, unless the call was aborted.
   */
  private handleError(
    error: unknown,
    attempts: ReadonlyArray<ModelRetryAttempt<LanguageModel>>,
    callOptions: LanguageModelCallOptions,
    inboundSignal: AbortSignal | undefined,
    currentRetry: Retry<LanguageModel> | undefined,
  ) {
    return evaluateError({
      abortSignal: inboundSignal,
      currentRetry,
      error,
      model: this.currentModel,
      options: callOptions,
      attempts,
      retries: this.options.retries,
      onError: this.options.onError,
      resolve: resolveLanguageModel,
    });
  }

  /**
   * Fire the `onFailure` callback for a terminally failed operation. The
   * final attempt (last entry of `attempts`) is surfaced as `current`.
   */
  private emitFailure(
    attempts: Array<ModelRetryAttempt<LanguageModel>>,
    error: unknown,
  ) {
    if (!this.options.onFailure) return;
    const current = attempts.at(-1);
    if (!current || !isErrorAttempt(current)) return;
    this.options.onFailure({ current, attempts, error });
  }

  async doGenerate(
    callOptions: LanguageModelCallOptions,
  ): Promise<LanguageModelResult> {
    /**
     * Resolve the starting model (base or sticky)
     */
    const startModel = this.resolveStartModel();
    this.currentModel = startModel;

    /**
     * If retries are disabled, bypass retry machinery entirely
     */
    if (this.isDisabled()) {
      return this.currentModel.doGenerate(callOptions);
    }

    const recorder = await createRetryTelemetry(this.telemetrySettings, {
      operation: 'doGenerate',
      genAiOperation: 'chat',
      provider: startModel.provider,
      modelId: startModel.modelId,
    });

    /**
     * Shared attempts array, threaded into `withRetry` so it stays populated
     * (including the final failed attempt) when the retry loop throws.
     */
    const attempts: Array<ModelRetryAttempt<LanguageModel>> = [];
    let operationError: unknown;
    /**
     * Only the retry loop is guarded. `onSuccess` runs after it, so a throwing
     * handler is not reported as an attempt failure — the request succeeded,
     * and firing `onFailure` too would report both outcomes for one request
     * with a stale attempt as `current`.
     */
    let result: LanguageModelResult;
    let finalCallOptions: LanguageModelCallOptions;
    try {
      const retried = await this.withRetry({
        fn: async (retryCallOptions) => {
          return this.currentModel.doGenerate(retryCallOptions);
        },
        callOptions: callOptions,
        attempts,
        recorder,
      });
      result = retried.result;
      finalCallOptions = retried.callOptions;
    } catch (error) {
      operationError = error;
      this.emitFailure(attempts, error);
      throw error;
    } finally {
      recorder?.endOperation({
        provider: this.currentModel.provider,
        modelId: this.currentModel.modelId,
        error: operationError,
      });
    }

    this.updateStickyModel(startModel);

    this.options.onSuccess?.({
      current: {
        type: 'success',
        model: this.currentModel,
        result,
        options: finalCallOptions,
      },
      attempts,
    });

    return result;
  }

  async doStream(
    callOptions: LanguageModelCallOptions,
  ): Promise<LanguageModelStream> {
    /**
     * Resolve the starting model (base or sticky)
     */
    const startModel = this.resolveStartModel();
    this.currentModel = startModel;

    /**
     * If retries are disabled, bypass retry machinery entirely
     */
    if (this.isDisabled()) {
      return this.currentModel.doStream(callOptions);
    }

    const recorder = await createRetryTelemetry(this.telemetrySettings, {
      operation: 'doStream',
      genAiOperation: 'chat',
      provider: startModel.provider,
      modelId: startModel.modelId,
    });

    /**
     * Perform the initial call to doStream with retry logic to handle errors before any data is streamed.
     */
    let result: LanguageModelStream;
    /**
     * Shared attempts array, threaded into `withRetry` so it stays populated
     * (including the final failed attempt) when the retry loop throws.
     */
    let attempts: Array<ModelRetryAttempt<LanguageModel>> = [];
    let finalCallOptions: LanguageModelCallOptions;
    /**
     * The open attempt span for the stream currently being consumed, closed
     * once its outcome (success, retry, or failure) is known.
     */
    let pendingAttempt: number | undefined;
    /**
     * The retry the stream being consumed runs under, for computing its call
     * options and judging its failures.
     */
    let currentRetry: Retry<LanguageModel> | undefined;
    try {
      const initial = await this.withRetry({
        fn: async (retryCallOptions) => {
          return this.currentModel.doStream(retryCallOptions);
        },
        callOptions: callOptions,
        attempts,
        recorder,
      });
      result = initial.result;
      attempts = initial.attempts;
      finalCallOptions = initial.callOptions;
      pendingAttempt = initial.pendingAttempt;
      currentRetry = initial.currentRetry;
    } catch (error) {
      /**
       * Every pre-stream attempt failed; record the operation failure before
       * the error propagates to the caller.
       */
      this.emitFailure(attempts, error);
      recorder?.endOperation({
        provider: this.currentModel.provider,
        modelId: this.currentModel.modelId,
        error,
      });
      throw error;
    }

    /**
     * Wrap the original stream to handle retries if an error occurs during
     * streaming, or if a `finish` part with a retryable finish reason is
     * received before any content has been forwarded downstream.
     */
    const retryableStream = new ReadableStream({
      start: async (controller) => {
        let reader:
          | ReadableStreamDefaultReader<LanguageModelStreamPart>
          | undefined;
        let isStreaming = false;
        /**
         * Whether a failure was forwarded to the consumer as a stream part.
         *
         * A stream that carries an `error` part still closes normally, so
         * completing the loop says nothing about whether the generation
         * succeeded. Without this the consumer sees a failure while
         * `onSuccess` reports a success.
         */
        let forwardedError = false;

        /** Set when the operation ends in failure, for the operation span. */
        let operationError: unknown;

        /**
         * Hand an unrecovered failure to the consumer. The call being aborted
         * rejects the stream, as the unwrapped model's stream would. Any other
         * failure is enqueued as an error part.
         *
         * `streamText` treats the two ways a model stream can fail
         * differently. A rejection with an abort error while its signal is
         * aborted runs its abort handling alone, and any other rejection
         * bypasses `onError`. An `error` part always reaches `onError`, so an
         * abort surfaced as a part reports a stop or a deadline as a failure
         * as well.
         */
        const surfaceError = (error: unknown) => {
          if (isCallAbort(error, callOptions.abortSignal, currentRetry)) {
            controller.error(error);
            return;
          }
          controller.enqueue({ type: 'error', error });
          controller.close();
        };
        try {
          while (true) {
            /**
             * Captured metadata from upstream stream parts, used to synthesize
             * a `LanguageModelResult` if a `finish` part triggers a retry
             * evaluation. Reset for each (re-)stream.
             */
            let capturedWarnings: LanguageModelResult['warnings'] = [];
            let capturedResponseMetadata: NonNullable<
              LanguageModelResult['response']
            > = {};

            /**
             * Buffer for the leading non-content parts (`stream-start`,
             * `response-metadata`, `text-start`, `reasoning-start`, …) of this
             * attempt. While no content has been forwarded the preamble is held
             * here rather than enqueued, so a pre-content retry can discard it
             * and the consumer sees exactly one preamble — the one belonging to
             * the model that actually produced the output. Reset per attempt;
             * flushed on the first content part or at completion.
             */
            let preambleBuffer: Array<LanguageModelStreamPart> = [];

            /**
             * Set when a `finish` part triggers a retry decision. Causes the
             * inner read loop to exit without enqueuing the finish part, and
             * the outer loop to re-stream against the next model.
             */
            let retryFromFinish: Retry<LanguageModel> | undefined;

            /** Unified finish reason of the last finish part seen this stream. */
            let streamFinishReason: string | undefined;

            try {
              reader = result.stream.getReader();

              while (true) {
                const { done, value } = await reader.read();
                if (done) break;

                /**
                 * If the stream part is an error and no data has been streamed yet, we can retry
                 * Throw the error to trigger the retry logic in withRetry
                 */
                if (value.type === 'error') {
                  if (!isStreaming) {
                    // If no data has been streamed yet, we can retry
                    throw value.error;
                  }
                  /**
                   * An abort is not forwarded as a part, for the same reason
                   * a surfaced one is not: it ends the attempt like a
                   * rejection does. Nothing the model sends after it, a later
                   * rejection included, may replace the abort.
                   */
                  if (
                    isCallAbort(
                      value.error,
                      callOptions.abortSignal,
                      currentRetry,
                    )
                  ) {
                    throw value.error;
                  }
                  /**
                   * Past the commit boundary the error belongs to the
                   * consumer's stream and cannot be retried, but it is still a
                   * failure and must not be reported as a success.
                   */
                  forwardedError = true;
                }

                /**
                 * Capture warnings and response metadata so they can be
                 * folded into a synthetic generate result if a `finish` part
                 * triggers a retry evaluation later in the stream.
                 */
                if (value.type === 'stream-start') {
                  capturedWarnings = value.warnings;
                }
                if (value.type === 'response-metadata') {
                  capturedResponseMetadata = {
                    ...capturedResponseMetadata,
                    ...(value.id !== undefined ? { id: value.id } : {}),
                    ...(value.modelId !== undefined
                      ? { modelId: value.modelId }
                      : {}),
                    ...(value.timestamp !== undefined
                      ? { timestamp: value.timestamp }
                      : {}),
                  };
                }

                if (value.type === 'finish') {
                  streamFinishReason = value.finishReason.unified;
                }

                /**
                 * If the stream part is a `finish` and no data has been
                 * streamed yet, evaluate retryables against a synthetic
                 * generate result built from the finish payload plus any
                 * metadata captured so far. If a retry model is selected,
                 * drop this finish part and re-stream. Once content has been
                 * forwarded, retry is unsafe and the finish part flows
                 * through unchanged.
                 */
                if (value.type === 'finish' && !isStreaming) {
                  const finishCallOptions = mergeLanguageModelCallOptions({
                    callOptions,
                    currentRetry,
                  });

                  const synthetic: LanguageModelResult = {
                    content: [],
                    finishReason: value.finishReason,
                    usage: value.usage,
                    warnings: capturedWarnings,
                    request: result.request,
                    response: {
                      ...capturedResponseMetadata,
                      ...result.response,
                    },
                    providerMetadata: value.providerMetadata,
                  };

                  const { retryModel, attempt } = await this.handleResult(
                    synthetic,
                    attempts,
                    finishCallOptions,
                  );

                  if (retryModel) {
                    /**
                     * If the inbound abort signal is already aborted and the
                     * chosen retry does not supply a fresh deadline, skip the
                     * retry and let the finish part flow downstream. Unlike
                     * the error path there is no underlying error to rethrow.
                     */
                    if (
                      !isRetryCancelled(callOptions.abortSignal, retryModel)
                    ) {
                      /**
                       * Only record the attempt once it is known to be
                       * retried. A finish that flows downstream is the
                       * successful outcome, not a failed attempt, so it stays
                       * out of the attempts list.
                       */
                      attempts.push(attempt);
                      retryFromFinish = retryModel;
                      break;
                    }
                  }
                }

                /**
                 * Mark that streaming has started once we receive actual
                 * content. On the first content part, flush this attempt's
                 * buffered preamble (in order) ahead of the content, then
                 * forward normally from here on.
                 */
                if (isStreamContentPart(value)) {
                  isStreaming = true;
                  for (const buffered of preambleBuffer) {
                    controller.enqueue(buffered);
                  }
                  preambleBuffer = [];
                  controller.enqueue(value);
                } else if (!isStreaming) {
                  /**
                   * Pre-content part: buffer it so a pre-content retry can
                   * replace it with the next attempt's preamble.
                   */
                  preambleBuffer.push(value);
                } else {
                  /**
                   * Content already flowing: forward directly.
                   */
                  controller.enqueue(value);
                }
              }

              if (retryFromFinish) {
                const calculatedDelay = resolveBackoffDelay(
                  retryFromFinish,
                  attempts,
                );

                if (pendingAttempt !== undefined) {
                  recorder?.endAttempt({
                    attempt: pendingAttempt,
                    outcome: 'retry',
                    finishReason: streamFinishReason,
                    delayMs: calculatedDelay,
                  });
                }

                await waitBeforeRetry(
                  calculatedDelay,
                  callOptions.abortSignal,
                  retryFromFinish,
                );

                this.currentModel = retryFromFinish.model;
                currentRetry = retryFromFinish;

                const retriedResult = await this.withRetry({
                  fn: async (retryCallOptions) => {
                    return this.currentModel.doStream(retryCallOptions);
                  },
                  callOptions: callOptions,
                  attempts,
                  currentRetry,
                  recorder,
                });

                /**
                 * Cancelling a reader whose stream has already errored (e.g.
                 * a mid-stream `controller.error`) rejects with that stored
                 * error. Swallow it: the retry already succeeded and that
                 * rejection must not abort the wrapped stream.
                 */
                await reader?.cancel().catch(() => {});

                result = retriedResult.result;
                attempts = retriedResult.attempts;
                finalCallOptions = retriedResult.callOptions;
                pendingAttempt = retriedResult.pendingAttempt;
                currentRetry = retriedResult.currentRetry;

                continue;
              }

              if (pendingAttempt !== undefined) {
                recorder?.endAttempt({
                  attempt: pendingAttempt,
                  outcome: 'success',
                  finishReason: streamFinishReason,
                });
              }
              /**
               * A stream that completes with no content part still has its
               * preamble buffered. Flush it so a zero-content completion emits
               * its `stream-start` (and any metadata/finish) before closing.
               */
              for (const buffered of preambleBuffer) {
                controller.enqueue(buffered);
              }
              preambleBuffer = [];
              controller.close();
              break;
            } catch (error) {
              /**
               * A failure while acting on a finish retry, in its backoff wait
               * or in a re-stream whose own retries ran out, is not this
               * stream failing. The re-stream has already evaluated and
               * recorded its attempts, so evaluating it again here would add
               * an attempt that never ran. Leave it to the outer handler,
               * which surfaces it.
               */
              if (retryFromFinish !== undefined) {
                throw error;
              }

              /**
               * Content has already been forwarded downstream, so a retry
               * would re-stream and duplicate output. Surface the error and
               * stop, the same outcome as an `error` part arriving after
               * content (which the read loop forwards rather than retrying).
               * Retry stays possible only before the commit point; past it the
               * caller owns the partial stream.
               */
              if (isStreaming) {
                if (pendingAttempt !== undefined) {
                  recorder?.endAttempt({
                    attempt: pendingAttempt,
                    outcome: 'failure',
                    error,
                  });
                }
                operationError = error;
                surfaceError(error);
                /**
                 * Stop the model, which may still be sending: an error that
                 * came as a part leaves its stream open. Cancelling a stream
                 * that already rejected fails with that rejection, which is
                 * already surfaced.
                 */
                await reader?.cancel(error).catch(() => {});
                return;
              }

              /**
               * Get the retry call options for the failed attempt
               */
              const retryCallOptions = mergeLanguageModelCallOptions({
                callOptions,
                currentRetry,
              });

              /**
               * Check if the error from the stream can be retried.
               */
              const { retryModel, attempt, finalError } =
                await this.handleError(
                  error,
                  attempts,
                  retryCallOptions,
                  callOptions.abortSignal,
                  currentRetry,
                );

              /**
               * Save the attempt
               */
              attempts.push(attempt);

              /**
               * No retry matched. Surface the error rather than throw it: a
               * throw would escape `start()` as a stream rejection, which
               * bypasses `streamText`'s `onError` for anything but an abort.
               */
              if (!retryModel) {
                if (pendingAttempt !== undefined) {
                  recorder?.endAttempt({
                    attempt: pendingAttempt,
                    outcome: 'failure',
                    error,
                  });
                }
                operationError = finalError;
                this.emitFailure(attempts, finalError);
                surfaceError(finalError);
                return;
              }

              /**
               * If the inbound abort signal is already aborted and the chosen
               * retry does not supply a fresh deadline, the retry would die
               * instantly with the same abort. Surface the error rather than
               * fire a misleading retry against a dead signal.
               */
              if (isRetryCancelled(callOptions.abortSignal, retryModel)) {
                if (pendingAttempt !== undefined) {
                  recorder?.endAttempt({
                    attempt: pendingAttempt,
                    outcome: 'failure',
                    error,
                  });
                }
                operationError = error;
                this.emitFailure(attempts, error);
                surfaceError(error);
                return;
              }

              const calculatedDelay = resolveBackoffDelay(retryModel, attempts);

              if (pendingAttempt !== undefined) {
                recorder?.endAttempt({
                  attempt: pendingAttempt,
                  outcome: 'retry',
                  error,
                  delayMs: calculatedDelay,
                });
              }

              await waitBeforeRetry(
                calculatedDelay,
                callOptions.abortSignal,
                retryModel,
              );

              this.currentModel = retryModel.model;
              currentRetry = retryModel;

              /**
               * Retry the request by calling doStream again.
               * This will create a new stream.
               */
              const retriedResult = await this.withRetry({
                fn: async (retryCallOptions) => {
                  return this.currentModel.doStream(retryCallOptions);
                },
                callOptions: callOptions,
                attempts,
                currentRetry,
                recorder,
              });

              /**
               * Cancel the previous reader and stream if we are retrying.
               * Cancelling a reader whose stream has already errored (e.g. a
               * mid-stream `controller.error`) rejects with that stored
               * error. Swallow it: the retry already succeeded and that
               * rejection must not abort the wrapped stream.
               */
              await reader?.cancel().catch(() => {});

              result = retriedResult.result;
              attempts = retriedResult.attempts;
              finalCallOptions = retriedResult.callOptions;
              pendingAttempt = retriedResult.pendingAttempt;
              currentRetry = retriedResult.currentRetry;
            } finally {
              reader?.releaseLock();
            }
          }

          /**
           * Stream finished — finalize sticky model and, if it finished
           * *well*, fire onSuccess. Deferred to here (rather than after the
           * initial withRetry resolves) so the final model and full attempts
           * list are observed, including any mid-stream retries.
           *
           * A stream that forwarded an error part reached this point too: it
           * closed normally, carrying the failure as cargo. That is not a
           * success, and neither is it an attempt failure the retry loop can
           * report — the consumer already has it, through `streamText`'s own
           * `onError`. So nothing fires.
           */
          this.updateStickyModel(startModel);

          if (forwardedError) {
            return;
          }

          this.options.onSuccess?.({
            current: {
              type: 'success',
              model: this.currentModel,
              result,
              options: finalCallOptions,
            },
            attempts,
          });
        } catch (error) {
          /**
           * Anything that escaped the loop without reaching one of its
           * terminal branches — most notably a re-stream whose own retries
           * are exhausted. Letting it escape `start()` would reject the
           * stream, which bypasses the consumer's `onError` entirely, so it
           * is surfaced like every other unrecovered failure.
           */
          operationError = error;
          this.emitFailure(attempts, error);
          surfaceError(error);
        } finally {
          recorder?.endOperation({
            provider: this.currentModel.provider,
            modelId: this.currentModel.modelId,
            error: operationError,
          });
        }
      },
    });

    return {
      ...result,
      stream: retryableStream,
    };
  }
}
