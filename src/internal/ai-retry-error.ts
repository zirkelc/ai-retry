import { AISDKError } from '@ai-sdk/provider';
import { RetryError } from 'ai';
import type { CallRetryAttempt } from '../call/types.js';
import type { AnyModel, ModelRetryAttempt } from '../types.js';

const marker = `ai-retry.error.AiRetryError`;
const symbol = Symbol.for(marker);

/**
 * An attempt of either retry layer, as the error carries it.
 */
export type AiRetryErrorAttempt =
  | ModelRetryAttempt<AnyModel>
  | CallRetryAttempt<AnyModel>;

/**
 * The `RetryError` this library fails with once more than one attempt was made,
 * at either retry layer, carrying the attempts.
 *
 * A retryable model can run inside a call-level retry, which re-runs the whole
 * call with its own fallbacks. The call level reads the attempts from here, so
 * it does not run a model again that the model level already tried. The SDK's
 * own `maxRetries` fails with a plain `RetryError` of the same name and reason,
 * so without this marker the two cannot be told apart.
 *
 * Still a `RetryError` in every way a consumer can check: `RetryError.isInstance`
 * reads a marker the base constructor sets, and `name`, `reason`, `errors` and
 * `lastError` are unchanged.
 *
 * Detect it with {@link AiRetryError.isInstance}, never with `instanceof`: an
 * app can load two copies of this package, and an error thrown by one is not an
 * instance of the other's class.
 */
export class AiRetryError extends RetryError {
  private readonly [symbol] = true;

  /**
   * Held in a private field and read through a getter, so the attempts, with
   * their prompts and results, stay out of what loggers and `JSON.stringify`
   * print for the error.
   */
  readonly #attempts: ReadonlyArray<AiRetryErrorAttempt>;

  constructor({
    message,
    reason,
    errors,
    attempts,
  }: {
    message: string;
    reason: RetryError['reason'];
    errors: Array<unknown>;
    attempts: ReadonlyArray<AiRetryErrorAttempt>;
  }) {
    super({ message, reason, errors });
    this.#attempts = attempts;
  }

  /** Every attempt, in order, ending with the one that failed last. */
  get attempts(): ReadonlyArray<AiRetryErrorAttempt> {
    return this.#attempts;
  }

  static override isInstance(error: unknown): error is AiRetryError {
    return AISDKError.hasMarker(error, marker);
  }
}
