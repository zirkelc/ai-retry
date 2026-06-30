/**
 * OpenTelemetry + Langfuse setup for the telemetry example.
 *
 * Import this module before anything that emits spans: starting the Node SDK
 * registers the global tracer and context manager that ai-retry and the AI
 * SDK write into. Without an active context manager, ai-retry's spans cannot
 * nest under the AI SDK's `ai.generateText` trace.
 *
 * In AI SDK v7 the SDK no longer emits OpenTelemetry spans on its own: its
 * OTel integration moved to `@ai-sdk/otel` and must be registered explicitly
 * via `registerTelemetry(new OpenTelemetry())`. ai-retry's own spans do not
 * need this (they talk to OpenTelemetry directly) — registering it is what
 * makes the AI SDK's `ai.generateText` / `ai.streamText` parent spans appear,
 * so ai-retry's attempts nest under them.
 */
import { OpenTelemetry } from '@ai-sdk/otel';
import { LangfuseSpanProcessor } from '@langfuse/otel';
// Langfuse ships its own AI SDK integration (`@langfuse/vercel-ai-sdk`) that
// renames AI SDK spans to GenAI-semantic names (invoke_agent / chat / ...).
// Swap the registration below for it if you prefer Langfuse-shaped traces:
//   import { LangfuseVercelAiSdkIntegration } from '@langfuse/vercel-ai-sdk';
import { NodeSDK } from '@opentelemetry/sdk-node';
import { registerTelemetry } from 'ai';

/** Load LANGFUSE_* and OPENAI_API_KEY from .env before reading them below. */
process.loadEnvFile();

export const langfuseSpanProcessor = new LangfuseSpanProcessor({
  /** Export each span as it ends — ideal for short-lived scripts. */
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
 * Register the AI SDK's OpenTelemetry integration so `generateText` /
 * `streamText` emit spans into the tracer the Node SDK just registered.
 * `usage` adds token-usage attributes to the trace.
 */
registerTelemetry(new OpenTelemetry({ usage: true }));
// Langfuse-shaped alternative:
// registerTelemetry(new LangfuseVercelAiSdkIntegration());
