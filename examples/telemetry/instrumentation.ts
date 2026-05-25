/**
 * OpenTelemetry + Langfuse setup for the telemetry example.
 *
 * Import this module before anything that emits spans: starting the Node SDK
 * registers the global tracer and context manager that ai-retry and the AI
 * SDK write into. Without an active context manager, ai-retry's spans cannot
 * nest under the AI SDK's `ai.generateText` trace.
 */
import { LangfuseSpanProcessor } from '@langfuse/otel';
import { NodeSDK } from '@opentelemetry/sdk-node';

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
