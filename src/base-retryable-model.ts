import { type ParsedReset, parseReset } from './parse-reset.js';
import type {
  EmbeddingModel,
  LanguageModel,
  RetryableModelOptions,
} from './types.js';

export abstract class BaseRetryableModel<
  MODEL extends LanguageModel | EmbeddingModel,
> {
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
