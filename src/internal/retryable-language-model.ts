import { delay } from '@ai-sdk/provider-utils';
import { BaseRetryableModel } from './base-retryable-model.js';
import { calculateExponentialBackoff } from './calculate-exponential-backoff.js';
import { countModelAttempts } from './count-model-attempts.js';
import { findRetryModel } from './find-retry-model.js';
import { resolveLanguageModel } from './resolve-model.js';
import { mergeLanguageModelCallOptions } from './merge-retry-call-options.js';
import { createRetryTelemetry, type RetryTelemetry } from './telemetry.js';
import { prepareRetryError } from './prepare-retry-error.js';
import type {
  LanguageModel,
  LanguageModelCallOptions,
  LanguageModelResult,
  LanguageModelStream,
  LanguageModelStreamPart,
  OnRetryOverrides,
  Retry,
  RetryAttempt,
  RetryContext,
  RetryErrorAttempt,
  RetryResultAttempt,
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
  readonly specificationVersion = 'v3';

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
    attempts?: Array<RetryAttempt<LanguageModel>>;
    currentRetry?: Retry<LanguageModel>;
    recorder?: RetryTelemetry;
  }): Promise<{
    result: RESULT;
    attempts: Array<RetryAttempt<LanguageModel>>;
    callOptions: LanguageModelCallOptions;
    /**
     * For stream results: the still-open attempt span number, to be closed by
     * the caller once the consumption outcome is known. Undefined otherwise.
     */
    pendingAttempt?: number;
  }> {
    /**
     * Track all attempts.
     */
    const attempts: Array<RetryAttempt<LanguageModel>> = input.attempts ?? [];

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
        const currentAttempt: RetryAttempt<LanguageModel> = {
          ...previousAttempt,
          model: this.currentModel,
        };

        /**
         * Create a shallow copy of the attempts for testing purposes
         */
        const updatedAttempts = [...attempts];

        const context: RetryContext<LanguageModel> = {
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
        timeoutMs: currentRetry?.timeout,
      });

      try {
        /**
         * Call the function that may need to be retried
         */
        const result = await input.fn(retryCallOptions);

        /**
         * Check if the result should trigger a retry (only for generate results, not streams)
         */
        if (isGenerateResult(result)) {
          const { retryModel, attempt } = await this.handleResult(
            result,
            attempts,
            retryCallOptions,
          );

          attempts.push(attempt);

          if (retryModel) {
            /**
             * Calculate exponential backoff delay based on the number of
             * attempts for this specific model: baseDelay * backoffFactor^attempts.
             */
            let calculatedDelay: number | undefined;
            if (retryModel.delay) {
              const modelAttemptsCount = countModelAttempts(
                retryModel.model,
                attempts,
              );
              calculatedDelay = calculateExponentialBackoff(
                retryModel.delay,
                retryModel.backoffFactor,
                modelAttemptsCount,
              );
            }

            input.recorder?.endAttempt({
              attempt: attemptNumber,
              outcome: 'retry',
              finishReason: result.finishReason.unified,
              delayMs: calculatedDelay,
            });

            if (calculatedDelay !== undefined) {
              await delay(calculatedDelay, {
                abortSignal: retryCallOptions.abortSignal,
              });
            }

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
        };
      } catch (error) {
        const { retryModel, attempt, finalError } = await this.handleError(
          error,
          attempts,
          retryCallOptions,
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
        if (
          input.callOptions.abortSignal?.aborted &&
          retryModel.timeout === undefined
        ) {
          input.recorder?.endAttempt({
            attempt: attemptNumber,
            outcome: 'failure',
            error,
          });
          throw error;
        }

        /**
         * Calculate exponential backoff delay based on the number of attempts
         * for this specific model: baseDelay * backoffFactor^attempts.
         */
        let calculatedDelay: number | undefined;
        if (retryModel.delay) {
          const modelAttemptsCount = countModelAttempts(
            retryModel.model,
            attempts,
          );
          calculatedDelay = calculateExponentialBackoff(
            retryModel.delay,
            retryModel.backoffFactor,
            modelAttemptsCount,
          );
        }

        input.recorder?.endAttempt({
          attempt: attemptNumber,
          outcome: 'retry',
          error,
          delayMs: calculatedDelay,
        });

        if (calculatedDelay !== undefined) {
          await delay(calculatedDelay, {
            abortSignal: retryCallOptions.abortSignal,
          });
        }

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
    attempts: ReadonlyArray<RetryAttempt<LanguageModel>>,
    callOptions: LanguageModelCallOptions,
  ) {
    const resultAttempt: RetryResultAttempt = {
      type: 'result',
      result: result,
      model: this.currentModel,
      options: callOptions,
    };

    /**
     * Save the current attempt
     */
    const updatedAttempts = [...attempts, resultAttempt];

    const context: RetryContext<LanguageModel> = {
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
   * is wrapped in a `RetryError`.
   */
  private async handleError(
    error: unknown,
    attempts: ReadonlyArray<RetryAttempt<LanguageModel>>,
    callOptions: LanguageModelCallOptions,
  ) {
    const errorAttempt: RetryErrorAttempt<LanguageModel> = {
      type: 'error',
      error: error,
      model: this.currentModel,
      options: callOptions,
    };

    const updatedAttempts = [...attempts, errorAttempt];

    const context: RetryContext<LanguageModel> = {
      current: errorAttempt,
      attempts: updatedAttempts,
    };

    this.options.onError?.(context);

    const retryModel = await findRetryModel(
      this.options.retries,
      context,
      resolveLanguageModel,
    );

    const finalError = retryModel
      ? undefined
      : updatedAttempts.length > 1
        ? prepareRetryError(error, updatedAttempts)
        : error;

    return { retryModel, attempt: errorAttempt, finalError };
  }

  /**
   * Fire the `onFailure` callback for a terminally failed operation. The
   * final attempt (last entry of `attempts`) is surfaced as `current`.
   */
  private emitFailure(
    attempts: Array<RetryAttempt<LanguageModel>>,
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

    const recorder = await createRetryTelemetry(
      this.options.experimental_telemetry,
      {
        operation: 'doGenerate',
        genAiOperation: 'chat',
        provider: startModel.provider,
        modelId: startModel.modelId,
      },
    );

    /**
     * Shared attempts array, threaded into `withRetry` so it stays populated
     * (including the final failed attempt) when the retry loop throws.
     */
    const attempts: Array<RetryAttempt<LanguageModel>> = [];
    let operationError: unknown;
    try {
      const { result, callOptions: finalCallOptions } = await this.withRetry({
        fn: async (retryCallOptions) => {
          return this.currentModel.doGenerate(retryCallOptions);
        },
        callOptions: callOptions,
        attempts,
        recorder,
      });

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

    const recorder = await createRetryTelemetry(
      this.options.experimental_telemetry,
      {
        operation: 'doStream',
        genAiOperation: 'chat',
        provider: startModel.provider,
        modelId: startModel.modelId,
      },
    );

    /**
     * Perform the initial call to doStream with retry logic to handle errors before any data is streamed.
     */
    let result: LanguageModelStream;
    /**
     * Shared attempts array, threaded into `withRetry` so it stays populated
     * (including the final failed attempt) when the retry loop throws.
     */
    let attempts: Array<RetryAttempt<LanguageModel>> = [];
    let finalCallOptions: LanguageModelCallOptions;
    /**
     * The open attempt span for the stream currently being consumed, closed
     * once its outcome (success, retry, or failure) is known.
     */
    let pendingAttempt: number | undefined;
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
     * Track the current retry model for computing call options in the stream handler
     */
    let currentRetry: Retry<LanguageModel> | undefined;

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

        /** Set when the operation ends in failure, for the operation span. */
        let operationError: unknown;
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

                  attempts.push(attempt);

                  if (retryModel) {
                    /**
                     * If the inbound abort signal is already aborted and the
                     * chosen retry does not supply a fresh deadline, skip the
                     * retry and let the finish part flow downstream. Unlike
                     * the error path there is no underlying error to rethrow.
                     */
                    const abortedNoTimeout =
                      callOptions.abortSignal?.aborted &&
                      retryModel.timeout === undefined;

                    if (!abortedNoTimeout) {
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
                /**
                 * Calculate exponential backoff delay based on the number of
                 * attempts for this specific model.
                 */
                let calculatedDelay: number | undefined;
                if (retryFromFinish.delay) {
                  const modelAttemptsCount = countModelAttempts(
                    retryFromFinish.model,
                    attempts,
                  );
                  calculatedDelay = calculateExponentialBackoff(
                    retryFromFinish.delay,
                    retryFromFinish.backoffFactor,
                    modelAttemptsCount,
                  );
                }

                if (pendingAttempt !== undefined) {
                  recorder?.endAttempt({
                    attempt: pendingAttempt,
                    outcome: 'retry',
                    finishReason: streamFinishReason,
                    delayMs: calculatedDelay,
                  });
                }

                if (calculatedDelay !== undefined) {
                  await delay(calculatedDelay, {
                    abortSignal: callOptions.abortSignal,
                  });
                }

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
                await this.handleError(error, attempts, retryCallOptions);

              /**
               * Save the attempt
               */
              attempts.push(attempt);

              /**
               * No retry matched. Surface the error as a stream part so
               * `streamText`'s `onError` fires for the consumer. Throwing
               * here would escape `start()` and become a stream rejection,
               * which silently bypasses `onError`.
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
                controller.enqueue({ type: 'error', error: finalError });
                controller.close();
                return;
              }

              /**
               * If the inbound abort signal is already aborted and the chosen
               * retry does not supply a fresh deadline, the retry would die
               * instantly with the same abort. Surface the error rather than
               * fire a misleading retry against a dead signal.
               */
              if (
                callOptions.abortSignal?.aborted &&
                retryModel.timeout === undefined
              ) {
                if (pendingAttempt !== undefined) {
                  recorder?.endAttempt({
                    attempt: pendingAttempt,
                    outcome: 'failure',
                    error,
                  });
                }
                operationError = error;
                this.emitFailure(attempts, error);
                controller.enqueue({ type: 'error', error });
                controller.close();
                return;
              }

              /**
               * Calculate exponential backoff delay based on the number of
               * attempts for this specific model: baseDelay * backoffFactor^attempts.
               */
              let calculatedDelay: number | undefined;
              if (retryModel.delay) {
                const modelAttemptsCount = countModelAttempts(
                  retryModel.model,
                  attempts,
                );
                calculatedDelay = calculateExponentialBackoff(
                  retryModel.delay,
                  retryModel.backoffFactor,
                  modelAttemptsCount,
                );
              }

              if (pendingAttempt !== undefined) {
                recorder?.endAttempt({
                  attempt: pendingAttempt,
                  outcome: 'retry',
                  error,
                  delayMs: calculatedDelay,
                });
              }

              if (calculatedDelay !== undefined) {
                await delay(calculatedDelay, {
                  abortSignal: retryCallOptions.abortSignal,
                });
              }

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
            } finally {
              reader?.releaseLock();
            }
          }

          /**
           * Stream completed successfully — finalize sticky model and fire
           * onSuccess. Deferred to here (rather than after the initial
           * withRetry resolves) so the final model and full attempts list
           * are observed, including any mid-stream retries.
           */
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
