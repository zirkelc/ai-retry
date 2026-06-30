import type { Attributes, Context, Span, Tracer } from '@opentelemetry/api';
import type { RetryTelemetrySettings } from '../types.js';

/**
 * Outcome of a single attempt, recorded as `ai_retry.attempt.outcome`.
 *
 * - `success` — the attempt produced the final result.
 * - `retry` — the attempt failed (error) or was rejected (result-based) and a
 *   fallback attempt was scheduled. Forward-looking: the retry has not run yet
 *   when this attempt closes.
 * - `failure` — the attempt failed and no further retry was possible.
 */
export type AttemptOutcome = 'success' | 'retry' | 'failure';

/**
 * Final state of the whole operation, recorded as `ai_retry.outcome` on the
 * operation span.
 */
export type OperationOutcome = 'success' | 'failure';

/** OpenTelemetry GenAI semantic-convention operation name. */
export type GenAiOperation = 'chat' | 'embeddings' | 'generate_content';

/**
 * Describes the operation being instrumented.
 */
export interface OperationInfo {
  /**
   * Method kind, e.g. `doGenerate`, `doStream`, `doEmbed`. Used for the span
   * name and the `ai_retry.operation` attribute.
   */
  operation: string;
  /** Standard `gen_ai.operation.name` value for the underlying model call. */
  genAiOperation: GenAiOperation;
  provider: string;
  modelId: string;
}

export interface AttemptStart {
  /** 1-based attempt index. */
  attempt: number;
  provider: string;
  modelId: string;
  /** Timeout budget for this attempt, when the retry config set one. */
  timeoutMs?: number;
}

export interface AttemptEnd {
  attempt: number;
  outcome: AttemptOutcome;
  /** Backoff delay scheduled before the next attempt, in milliseconds. */
  delayMs?: number;
  /** Unified finish reason, for result-based outcomes. */
  finishReason?: string;
  /** The error that ended the attempt, if any. */
  error?: unknown;
}

export interface OperationEnd {
  /** The model that produced the final outcome. */
  provider: string;
  modelId: string;
  /** The error that ended the operation, if it failed. */
  error?: unknown;
}

/**
 * Sink-agnostic recorder driven by the retry loop. The retry code only ever
 * sees this interface; the concrete sink(s) behind it (today: OpenTelemetry)
 * decide how to materialize the data.
 */
export interface RetryTelemetry {
  startAttempt(input: AttemptStart): void;
  endAttempt(input: AttemptEnd): void;
  endOperation(input: OperationEnd): void;
}

/**
 * A consumer of normalized telemetry events. Additional sinks (e.g. the AI
 * SDK telemetry diagnostic channel, once it ships in a stable release) can be
 * added without touching the retry loop.
 */
interface TelemetrySink {
  operationStart(info: OperationInfo): void;
  attemptStart(input: AttemptStart): void;
  attemptEnd(input: AttemptEnd): void;
  operationEnd(input: OperationEnd): void;
}

/**
 * Fans normalized events out to every configured sink.
 */
class CompositeRetryTelemetry implements RetryTelemetry {
  #sinks: ReadonlyArray<TelemetrySink>;

  constructor(sinks: ReadonlyArray<TelemetrySink>) {
    this.#sinks = sinks;
  }

  startAttempt(input: AttemptStart): void {
    for (const sink of this.#sinks) sink.attemptStart(input);
  }

  endAttempt(input: AttemptEnd): void {
    for (const sink of this.#sinks) sink.attemptEnd(input);
  }

  endOperation(input: OperationEnd): void {
    for (const sink of this.#sinks) sink.operationEnd(input);
  }
}

/**
 * The lazily imported `@opentelemetry/api` module. Importing it only when
 * telemetry is enabled keeps the dependency genuinely optional: a consumer
 * that never opts in never loads it.
 */
type OtelApi = typeof import('@opentelemetry/api');
let otelApiPromise: Promise<OtelApi> | undefined;

function loadOtelApi(): Promise<OtelApi> {
  otelApiPromise ??= import('@opentelemetry/api').catch((cause: unknown) => {
    /** Reset so a later call can retry once the dependency is installed. */
    otelApiPromise = undefined;
    throw new Error(
      "ai-retry: telemetry is enabled but the optional peer dependency '@opentelemetry/api' is not installed. Install it with: npm install @opentelemetry/api",
      { cause },
    );
  });
  return otelApiPromise;
}

/**
 * Normalize an unknown thrown value into the fields recorded on error spans:
 * - `name`: the error class name (e.g. `AI_APICallError`, `TimeoutError`), or
 *   `typeof` for non-`Error` values.
 * - `message`: the error message.
 * - `status`: HTTP status code, when the error carries one (e.g. `APICallError`).
 */
function describeError(error: unknown): {
  name: string;
  message: string;
  status?: number;
} {
  const name = error instanceof Error ? error.name : typeof error;
  const message = error instanceof Error ? error.message : String(error);
  const status =
    typeof error === 'object' &&
    error !== null &&
    'statusCode' in error &&
    typeof error.statusCode === 'number'
      ? error.statusCode
      : undefined;
  return { name, message, status };
}

/**
 * Translates retry telemetry events into OpenTelemetry spans.
 *
 * The operation span is created against the currently active context, so it
 * nests under any surrounding span (notably the AI SDK's
 * `ai.generateText.doGenerate`). Attempt spans are created against the
 * operation's context, giving a parent → attempts tree.
 */
class OpenTelemetrySink implements TelemetrySink {
  #api: OtelApi;
  #tracer: Tracer;
  #settings: RetryTelemetrySettings;
  #operationSpan?: Span;
  #operationContext?: Context;
  #attemptSpans = new Map<number, Span>();
  #attemptCount = 0;

  constructor(api: OtelApi, settings: RetryTelemetrySettings) {
    this.#api = api;
    this.#settings = settings;
    this.#tracer = settings.tracer ?? api.trace.getTracer('ai-retry');
  }

  operationStart(info: OperationInfo): void {
    const { trace, context } = this.#api;

    const attributes: Attributes = {
      'ai_retry.operation': info.operation,
      'ai_retry.model.start': `${info.provider}/${info.modelId}`,
      'gen_ai.operation.name': info.genAiOperation,
    };
    if (this.#settings.metadata) {
      for (const [key, value] of Object.entries(this.#settings.metadata)) {
        attributes[`ai_retry.metadata.${key}`] = value;
      }
    }

    this.#operationSpan = this.#tracer.startSpan(
      `ai_retry.${info.operation}`,
      { attributes },
      context.active(),
    );
    this.#operationContext = trace.setSpan(
      context.active(),
      this.#operationSpan,
    );
  }

  attemptStart({ attempt, provider, modelId, timeoutMs }: AttemptStart): void {
    this.#attemptCount = Math.max(this.#attemptCount, attempt);

    const attributes: Attributes = {
      'ai_retry.attempt.number': attempt,
      'ai_retry.attempt.model': `${provider}/${modelId}`,
      /** Standard gen_ai conventions: keep tools like Langfuse exporting and rendering attempt spans. */
      'gen_ai.request.model': modelId,
      'gen_ai.provider.name': provider,
    };
    if (timeoutMs !== undefined) {
      attributes['ai_retry.attempt.timeout_ms'] = timeoutMs;
    }

    const span = this.#tracer.startSpan(
      'ai_retry.attempt',
      { attributes },
      this.#operationContext,
    );
    this.#attemptSpans.set(attempt, span);
  }

  attemptEnd({
    attempt,
    outcome,
    delayMs,
    finishReason,
    error,
  }: AttemptEnd): void {
    const span = this.#attemptSpans.get(attempt);
    if (!span) return;

    span.setAttribute('ai_retry.attempt.outcome', outcome);
    /**
     * Whether the attempt ended with a model result or a thrown error. An
     * attempt is one or the other, so the presence of `error` decides it.
     */
    span.setAttribute(
      'ai_retry.attempt.type',
      error !== undefined ? 'error' : 'result',
    );
    if (delayMs !== undefined) {
      span.setAttribute('ai_retry.attempt.delay_ms', delayMs);
    }
    if (finishReason !== undefined) {
      span.setAttribute('ai_retry.attempt.finish_reason', finishReason);
      span.setAttribute('gen_ai.response.finish_reasons', [finishReason]);
    }
    if (error !== undefined) {
      this.#recordError(span, error, 'ai_retry.attempt.error');
    }

    span.end();
    this.#attemptSpans.delete(attempt);
  }

  operationEnd({ provider, modelId, error }: OperationEnd): void {
    const span = this.#operationSpan;
    if (!span) return;

    span.setAttribute('ai_retry.attempts', this.#attemptCount);
    span.setAttribute('ai_retry.model.final', `${provider}/${modelId}`);
    span.setAttribute(
      'ai_retry.outcome',
      error !== undefined ? 'failure' : 'success',
    );
    if (error !== undefined) this.#recordError(span, error, 'ai_retry.error');

    span.end();
    this.#operationSpan = undefined;
  }

  #recordError(span: Span, error: unknown, errorAttribute: string): void {
    const exception = error instanceof Error ? error : new Error(String(error));

    const info = describeError(error);
    span.setAttribute(`${errorAttribute}.name`, info.name);
    span.setAttribute(`${errorAttribute}.message`, info.message);
    if (info.status !== undefined) {
      span.setAttribute(`${errorAttribute}.status`, info.status);
    }

    /** One level of the error chain — the underlying cause, when present. */
    const cause = error instanceof Error ? error.cause : undefined;
    if (cause !== undefined) {
      const causeInfo = describeError(cause);
      span.setAttribute(`${errorAttribute}.cause.name`, causeInfo.name);
      span.setAttribute(`${errorAttribute}.cause.message`, causeInfo.message);
      if (causeInfo.status !== undefined) {
        span.setAttribute(`${errorAttribute}.cause.status`, causeInfo.status);
      }
    }

    span.recordException(exception);
    span.setStatus({
      code: this.#api.SpanStatusCode.ERROR,
      message: info.message,
    });
  }
}

/**
 * Build a telemetry recorder for one operation, or `undefined` when telemetry
 * is disabled. The retry loop treats `undefined` as a no-op.
 *
 * Resolving the optional `@opentelemetry/api` dependency is deferred to here,
 * so it is only loaded when telemetry is actually enabled.
 */
export async function createRetryTelemetry(
  settings: RetryTelemetrySettings | undefined,
  info: OperationInfo,
): Promise<RetryTelemetry | undefined> {
  if (!settings?.isEnabled) return undefined;

  const api = await loadOtelApi();

  const sinks: Array<TelemetrySink> = [new OpenTelemetrySink(api, settings)];

  for (const sink of sinks) sink.operationStart(info);

  return new CompositeRetryTelemetry(sinks);
}
