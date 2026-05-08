import { delay } from '@ai-sdk/provider-utils';
import { BaseRetryableModel } from './base-retryable-model.js';
import { calculateExponentialBackoff } from './calculate-exponential-backoff.js';
import { countModelAttempts } from './count-model-attempts.js';
import { findRetryModel } from './find-retry-model.js';
import { mergeLanguageModelCallOptions } from './merge-retry-call-options.js';
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
import { isGenerateResult, isStreamContentPart } from './guards.js';

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
  }): Promise<{
    result: RESULT;
    attempts: Array<RetryAttempt<LanguageModel>>;
    callOptions: LanguageModelCallOptions;
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
            if (retryModel.delay) {
              /**
               * Calculate exponential backoff delay based on the number of attempts for this specific model.
               * The delay grows exponentially: baseDelay * backoffFactor^attempts
               * Example: With delay=1000ms and backoffFactor=2:
               * - Attempt 1: 1000ms
               * - Attempt 2: 2000ms
               * - Attempt 3: 4000ms
               */
              const modelAttemptsCount = countModelAttempts(
                retryModel.model,
                attempts,
              );
              const calculatedDelay = calculateExponentialBackoff(
                retryModel.delay,
                retryModel.backoffFactor,
                modelAttemptsCount,
              );
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
        }

        return { result, attempts, callOptions: retryCallOptions };
      } catch (error) {
        const { retryModel, attempt, finalError } = await this.handleError(
          error,
          attempts,
          retryCallOptions,
        );

        attempts.push(attempt);

        if (!retryModel) {
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
          throw error;
        }

        if (retryModel.delay) {
          /**
           * Calculate exponential backoff delay based on the number of attempts for this specific model.
           * The delay grows exponentially: baseDelay * backoffFactor^attempts
           */
          const modelAttemptsCount = countModelAttempts(
            retryModel.model,
            attempts,
          );
          const calculatedDelay = calculateExponentialBackoff(
            retryModel.delay,
            retryModel.backoffFactor,
            modelAttemptsCount,
          );
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

    const retryModel = await findRetryModel(this.options.retries, context);

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

    const retryModel = await findRetryModel(this.options.retries, context);

    const finalError = retryModel
      ? undefined
      : updatedAttempts.length > 1
        ? prepareRetryError(error, updatedAttempts)
        : error;

    return { retryModel, attempt: errorAttempt, finalError };
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

    const {
      result,
      attempts,
      callOptions: finalCallOptions,
    } = await this.withRetry({
      fn: async (retryCallOptions) => {
        return this.currentModel.doGenerate(retryCallOptions);
      },
      callOptions: callOptions,
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

    /**
     * Perform the initial call to doStream with retry logic to handle errors before any data is streamed.
     */
    let {
      result,
      attempts,
      callOptions: finalCallOptions,
    } = await this.withRetry({
      fn: async (retryCallOptions) => {
        return this.currentModel.doStream(retryCallOptions);
      },
      callOptions: callOptions,
    });

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
           * Set when a `finish` part triggers a retry decision. Causes the
           * inner read loop to exit without enqueuing the finish part, and
           * the outer loop to re-stream against the next model.
           */
          let retryFromFinish: Retry<LanguageModel> | undefined;

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
               * Mark that streaming has started once we receive actual content
               */
              if (isStreamContentPart(value)) {
                isStreaming = true;
              }

              /**
               * Enqueue the chunk to the consumer of the stream
               */
              controller.enqueue(value);
            }

            if (retryFromFinish) {
              if (retryFromFinish.delay) {
                /**
                 * Calculate exponential backoff delay based on the number
                 * of attempts for this specific model.
                 */
                const modelAttemptsCount = countModelAttempts(
                  retryFromFinish.model,
                  attempts,
                );
                const calculatedDelay = calculateExponentialBackoff(
                  retryFromFinish.delay,
                  retryFromFinish.backoffFactor,
                  modelAttemptsCount,
                );
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
              });

              await reader?.cancel();

              result = retriedResult.result;
              attempts = retriedResult.attempts;
              finalCallOptions = retriedResult.callOptions;

              continue;
            }

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
            const { retryModel, attempt, finalError } = await this.handleError(
              error,
              attempts,
              retryCallOptions,
            );

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
              controller.enqueue({ type: 'error', error });
              controller.close();
              return;
            }

            if (retryModel.delay) {
              /**
               * Calculate exponential backoff delay based on the number of attempts for this specific model.
               * The delay grows exponentially: baseDelay * backoffFactor^attempts
               */
              const modelAttemptsCount = countModelAttempts(
                retryModel.model,
                attempts,
              );
              const calculatedDelay = calculateExponentialBackoff(
                retryModel.delay,
                retryModel.backoffFactor,
                modelAttemptsCount,
              );
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
            });

            /**
             * Cancel the previous reader and stream if we are retrying
             */
            await reader?.cancel();

            result = retriedResult.result;
            attempts = retriedResult.attempts;
            finalCallOptions = retriedResult.callOptions;
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
      },
    });

    return {
      ...result,
      stream: retryableStream,
    };
  }
}
