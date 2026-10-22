/**
 * The OpenTelemetry and Langfuse setup for the telemetry example.
 *
 * Import this module before all code that emits spans. When the Node SDK
 * starts, it registers the global tracer and the context manager that ai-retry
 * and the AI SDK use. Without an active context manager, the ai-retry spans
 * cannot be child spans of the `ai.generateText` span.
 *
 * In AI SDK v7, the AI SDK does not emit OpenTelemetry spans by default. You
 * must register `@ai-sdk/otel` with `registerTelemetry(new OpenTelemetry())`.
 * The ai-retry spans do not need it, because ai-retry uses OpenTelemetry
 * directly. It adds the `ai.generateText` and `ai.streamText` parent spans.
 */
import { OpenTelemetry } from '@ai-sdk/otel';
import { LangfuseSpanProcessor } from '@langfuse/otel';
// Langfuse has its own AI SDK integration (`@langfuse/vercel-ai-sdk`), which
// gives the AI SDK spans GenAI names (invoke_agent, chat, ...). To use it,
// replace the registration at the end of this file:
//   import { LangfuseVercelAiSdkIntegration } from '@langfuse/vercel-ai-sdk';
import { NodeSDK } from '@opentelemetry/sdk-node';
import { registerTelemetry } from 'ai';

/** Load the LANGFUSE_* variables from `.env`. */
process.loadEnvFile();

export const langfuseSpanProcessor = new LangfuseSpanProcessor({
  /** Export each span when it ends, for a script that runs only a short time. */
  exportMode: 'immediate',
  publicKey: process.env.LANGFUSE_PUBLIC_KEY,
  secretKey: process.env.LANGFUSE_SECRET_KEY,
  baseUrl: process.env.LANGFUSE_BASE_URL ?? 'https://cloud.langfuse.com',
  environment: process.env.STAGE ?? 'development',
});

const sdk = new NodeSDK({
  spanProcessors: [langfuseSpanProcessor],
});

sdk.start();

/**
 * Register the OpenTelemetry integration of the AI SDK, so that `generateText`
 * and `streamText` emit spans to the tracer of the Node SDK. `usage` adds
 * token usage attributes to the trace.
 */
registerTelemetry(new OpenTelemetry({ usage: true }));
// The Langfuse alternative:
// registerTelemetry(new LangfuseVercelAiSdkIntegration());
