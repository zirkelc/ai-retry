import { type ParsedReset, parseReset } from './parse-reset.js';
import { isErrorAttempt } from './guards.js';
import { waitBeforeRetry } from './retry-signal.js';
import type {
  AnyModel,
  ModelFailureContext,
  ModelRetryAttempt,
  RetryableModelOptions,
  RetryTimeout,
} from '../types.js';

/**
 * Why an operation stopped, decided where its retry loop stopped rather than
 * guessed afterwards from the error it throws. Only that point knows whether
 * the loop gave up because the call was aborted: a retry that would run
 * against a dead signal, or a cancel while waiting for one, stops the loop
 * whatever error the last attempt failed with.
 */
export type RetryStop = {
  /** Whether the operation ended because the call was aborted. */
  aborted: boolean;
};

export abstract class BaseRetryableModel<MODEL extends AnyModel> {
  protected baseModel: MODEL;
  protected currentModel: MODEL;
  protected options: RetryableModelOptions<MODEL>;

  private parsedReset: ParsedReset;

  /** The model that last succeeded via retry, used for subsequent requests. */
  private stickyState?: {
    model: MODEL;
    setAt: number;
    requestsRemaining: number;
  };

  constructor(options: RetryableModelOptions<MODEL>) {
    this.options = options;
    this.baseModel = options.model;
    this.currentModel = options.model;
    this.parsedReset = parseReset(options.reset ?? `after-request`);
  }

  /**
   * Determine which model to start the request with,
   * considering the sticky model and reset policy.
   */
  protected resolveStartModel(): MODEL {
    if (!this.stickyState) {
      return this.baseModel;
    }

    if (this.parsedReset.type === `requests`) {
      if (this.stickyState.requestsRemaining > 0) {
        this.stickyState.requestsRemaining--;
        return this.stickyState.model;
      }
    } else {
      const elapsed = Date.now() - this.stickyState.setAt;
      if (elapsed < this.parsedReset.count * 1_000) {
        return this.stickyState.model;
      }
    }

    this.stickyState = undefined;
    return this.baseModel;
  }

  /**
   * After a successful request, update sticky model if a retry occurred.
   */
  protected updateStickyModel(startModel: MODEL): void {
    if (this.currentModel !== startModel) {
      this.stickyState = {
        model: this.currentModel,
        setAt: Date.now(),
        requestsRemaining:
          this.parsedReset.type === `requests` ? this.parsedReset.count : 0,
      };
    }
  }

  /**
   * Resolve the telemetry settings, preferring `telemetry` over the deprecated
   * `experimental_telemetry` alias.
   */
  protected get telemetrySettings(): RetryableModelOptions<MODEL>['telemetry'] {
    return this.options.telemetry ?? this.options.experimental_telemetry;
  }

  /**
   * Report a terminally failed operation. The final attempt (last entry of
   * `attempts`) is surfaced as `current`, so it has to be an error attempt: an
   * operation that ended some other way, such as a callback throwing, has no
   * failed attempt to report and reports nothing.
   */
  protected emitFailure(
    attempts: Array<ModelRetryAttempt<MODEL>>,
    error: unknown,
    stop: RetryStop,
    onFailure:
      | ((context: ModelFailureContext<MODEL>) => void)
      | undefined = this.options.onFailure,
  ): void {
    if (!onFailure) return;
    const current = attempts.at(-1);
    if (!current || !isErrorAttempt(current)) return;
    onFailure({
      current,
      attempts,
      error,
      aborted: stop.aborted,
    } as unknown as ModelFailureContext<MODEL>);
  }

  /**
   * Wait out the backoff delay before the next attempt. A cancel during the
   * wait stops the operation between attempts, so it is recorded here: as the
   * reason the loop stopped, and as an error attempt against the model and
   * options of the attempt it followed, which the failure then reports.
   */
  protected async waitBeforeNextAttempt(
    delayMs: number | undefined,
    inboundSignal: AbortSignal | undefined,
    retry: { timeout?: RetryTimeout } | undefined,
    attempts: Array<ModelRetryAttempt<MODEL>>,
    stop: RetryStop,
  ): Promise<void> {
    try {
      await waitBeforeRetry(delayMs, inboundSignal, retry);
    } catch (error) {
      stop.aborted = true;
      const last = attempts.at(-1);
      if (last) {
        attempts.push({
          type: 'error',
          error,
          model: last.model,
          options: last.options,
        } as ModelRetryAttempt<MODEL>);
      }
      throw error;
    }
  }

  /**
   * Check if retries are disabled
   */
  protected isDisabled(): boolean {
    if (this.options.disabled === undefined) {
      return false;
    }

    return typeof this.options.disabled === `function`
      ? this.options.disabled()
      : this.options.disabled;
  }
}
