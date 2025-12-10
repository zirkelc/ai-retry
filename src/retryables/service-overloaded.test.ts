import { anthropic } from '@ai-sdk/anthropic';
import { openai } from '@ai-sdk/openai';
import {
  convertArrayToReadableStream,
  convertAsyncIterableToArray,
} from '@ai-sdk/provider-utils/test';
import { createTestServer } from '@ai-sdk/test-server';
import { APICallError, embed, generateText, streamText } from 'ai';
import { describe, expect, it, vi } from 'vitest';
import { createRetryable } from '../create-retryable-model.js';
import {
  chunksToText,
  MockEmbeddingModel,
  MockLanguageModel,
} from '../test-utils.js';
import type {
  EmbeddingModelEmbed,
  LanguageModelGenerate,
  LanguageModelStreamPart,
} from '../types.js';
import { serviceOverloaded } from './service-overloaded.js';

const mockResultText = 'Hello, world!';

const mockResult: LanguageModelGenerate = {
  finishReason: 'stop',
  usage: {
    inputTokens: { total: 10, noCache: 0, cacheRead: 0, cacheWrite: 0 },
    outputTokens: { total: 20, text: 0, reasoning: 0 },
  },
  content: [{ type: 'text', text: mockResultText }],
  warnings: [],
};

const overloadedError = new APICallError({
  message: 'Overloaded',
  url: '',
  requestBodyValues: {},
  statusCode: 529,
  responseHeaders: {},
  responseBody: '',
  isRetryable: false,
  data: {},
});

const mockStreamChunks: LanguageModelStreamPart[] = [
  {
    type: 'stream-start',
    warnings: [],
  },
  {
    type: 'response-metadata',
    id: 'id-0',
    modelId: 'mock-model-id',
    timestamp: new Date(0),
  },
  { type: 'text-start', id: '1' },
  { type: 'text-delta', id: '1', delta: 'Hello' },
  { type: 'text-delta', id: '1', delta: ', ' },
  { type: 'text-delta', id: '1', delta: 'world!' },
  { type: 'text-end', id: '1' },
  {
    type: 'finish',
    finishReason: 'stop',
    usage: {
      inputTokens: { total: 10, noCache: 0, cacheRead: 0, cacheWrite: 0 },
      outputTokens: { total: 20, text: 0, reasoning: 0 },
    },
  },
];

const errorChunk = (error: Record<string, any>): string =>
  `data: {"type":"error","error":${JSON.stringify(error)}}\n\n`;
const textChunks = {
  openai: (deltas: string[]): string[] => [
    `data:{"type":"response.created","response":{"id":"resp_67c9a81b6a048190a9ee441c5755a4e8","object":"response","created_at":1741269019,"status":"in_progress","error":null,"incomplete_details":null,"input":[],"instructions":null,"max_output_tokens":null,"model":"gpt-4o-2024-07-18","output":[],"parallel_tool_calls":true,"previous_response_id":null,"reasoning":{"effort":null,"summary":null},"store":true,"temperature":0.3,"text":{"format":{"type":"text"}},"tool_choice":"auto","tools":[],"top_p":1,"truncation":"disabled","usage":null,"user":null,"metadata":{}}}\n\n`,
    `data:{"type":"response.in_progress","response":{"id":"resp_67c9a81b6a048190a9ee441c5755a4e8","object":"response","created_at":1741269019,"status":"in_progress","error":null,"incomplete_details":null,"input":[],"instructions":null,"max_output_tokens":null,"model":"gpt-4o-2024-07-18","output":[],"parallel_tool_calls":true,"previous_response_id":null,"reasoning":{"effort":null,"summary":null},"store":true,"temperature":0.3,"text":{"format":{"type":"text"}},"tool_choice":"auto","tools":[],"top_p":1,"truncation":"disabled","usage":null,"user":null,"metadata":{}}}\n\n`,
    `data:{"type":"response.output_item.added","output_index":0,"item":{"id":"msg_67c9a81dea8c8190b79651a2b3adf91e","type":"message","status":"in_progress","role":"assistant","content":[]}}\n\n`,
    `data:{"type":"response.content_part.added","item_id":"msg_67c9a81dea8c8190b79651a2b3adf91e","output_index":0,"content_index":0,"part":{"type":"output_text","text":"","annotations":[],"logprobs": []}}\n\n`,
    ...deltas.map(
      (delta) =>
        `data:{"type":"response.output_text.delta","item_id":"msg_67c9a81dea8c8190b79651a2b3adf91e","output_index":0,"content_index":0,"delta":${JSON.stringify(delta)},"logprobs": []}\n\n`,
    ),
    `data:{"type":"response.output_text.done","item_id":"msg_67c9a8787f4c8190b49c858d4c1cf20c","output_index":0,"content_index":0,"text":"Hello, World!"}\n\n`,
    `data:{"type":"response.content_part.done","item_id":"msg_67c9a8787f4c8190b49c858d4c1cf20c","output_index":0,"content_index":0,"part":{"type":"output_text","text":"Hello, World!","annotations":[],"logprobs": []}}\n\n`,
    `data:{"type":"response.output_item.done","output_index":0,"item":{"id":"msg_67c9a8787f4c8190b49c858d4c1cf20c","type":"message","status":"completed","role":"assistant","content":[{"type":"output_text","text":"Hello, World!","annotations":[],"logprobs": []}]}}\n\n`,
    `data:{"type":"response.completed","response":{"id":"resp_67c9a878139c8190aa2e3105411b408b","object":"response","created_at":1741269112,"status":"completed","error":null,"incomplete_details":null,"input":[],"instructions":null,"max_output_tokens":null,"model":"gpt-4o-2024-07-18","output":[{"id":"msg_67c9a8787f4c8190b49c858d4c1cf20c","type":"message","status":"completed","role":"assistant","content":[{"type":"output_text","text":"Hello, World!","annotations":[]}]}],"parallel_tool_calls":true,"previous_response_id":null,"reasoning":{"effort":null,"summary":null},"store":true,"temperature":0.3,"text":{"format":{"type":"text"}},"tool_choice":"auto","tools":[],"top_p":1,"truncation":"disabled","usage":{"input_tokens":543,"input_tokens_details":{"cached_tokens":234},"output_tokens":478,"output_tokens_details":{"reasoning_tokens":123},"total_tokens":512},"user":null,"metadata":{}}}\n\n`,
  ],
};

const mockEmbeddings: EmbeddingModelEmbed = {
  embeddings: [[0.1, 0.2, 0.3]],
  warnings: [],
};

describe('serviceOverloaded', () => {
  describe('generateText', () => {
    it('should succeed without errors', async () => {
      // Arrange
      const baseModel = new MockLanguageModel({ doGenerate: mockResult });
      const retryModel = new MockLanguageModel({ doGenerate: mockResult });

      // Act
      const result = await generateText({
        model: createRetryable({
          model: baseModel,
          retries: [serviceOverloaded(retryModel)],
        }),
        prompt: 'Hello!',
      });

      // Assert
      expect(baseModel.doGenerate).toHaveBeenCalledTimes(1);
      expect(result.text).toBe(mockResultText);
    });

    it('should retry for status 529', async () => {
      // Arrange
      const baseModel = new MockLanguageModel({ doGenerate: overloadedError });
      const retryModel = new MockLanguageModel({ doGenerate: mockResult });

      // Act
      const result = await generateText({
        model: createRetryable({
          model: baseModel,
          retries: [serviceOverloaded(retryModel)],
        }),
        prompt: 'Hello!',
      });

      // Assert
      expect(baseModel.doGenerate).toHaveBeenCalledTimes(1);
      expect(retryModel.doGenerate).toHaveBeenCalledTimes(1);
      expect(result.text).toBe(mockResultText);
    });

    it('should not retry for status 200', async () => {
      // Arrange
      const baseModel = new MockLanguageModel({ doGenerate: mockResult });
      const retryModel = new MockLanguageModel({ doGenerate: mockResult });

      // Act
      const result = await generateText({
        model: createRetryable({
          model: baseModel,
          retries: [serviceOverloaded(retryModel)],
        }),
        prompt: 'Hello!',
      });

      // Assert
      expect(baseModel.doGenerate).toHaveBeenCalledTimes(1);
      expect(retryModel.doGenerate).toHaveBeenCalledTimes(0);
      expect(result.text).toBe(mockResultText);
    });

    it('should not retry if no matches', async () => {
      // Arrange
      const baseModel = new MockLanguageModel({
        doGenerate: new APICallError({
          message: 'Some other error',
          url: '',
          requestBodyValues: {},
          statusCode: 400,
          responseHeaders: {},
          responseBody: '{}',
          isRetryable: false,
          data: {
            error: {
              message: 'Some other error',
              type: null,
              param: 'prompt',
              code: 'other_error',
            },
          },
        }),
      });

      const retryModel = new MockLanguageModel({ doGenerate: mockResult });

      // Act
      const result = generateText({
        model: createRetryable({
          model: baseModel,
          retries: [serviceOverloaded(retryModel)],
        }),
        prompt: 'Hello!',
        maxRetries: 0,
      });

      // Assert
      await expect(result).rejects.toThrowError(APICallError);
      expect(baseModel.doGenerate).toHaveBeenCalledTimes(1);
      expect(retryModel.doGenerate).toHaveBeenCalledTimes(0);
    });
  });

  describe('streamText', () => {
    it('should succeed without errors', async () => {
      // Arrange
      const baseModel = new MockLanguageModel({
        doStream: {
          stream: convertArrayToReadableStream(mockStreamChunks),
        },
      });
      const retryModel = new MockLanguageModel({
        doStream: {
          stream: convertArrayToReadableStream(mockStreamChunks),
        },
      });
      let error: unknown;

      // Act
      const result = streamText({
        model: createRetryable({
          model: baseModel,
          retries: [serviceOverloaded(retryModel)],
        }),
        prompt: 'Hello!',
        onError(data) {
          error = data.error;
        },
      });

      const chunks = await convertAsyncIterableToArray(result.fullStream);

      // Assert
      expect(baseModel.doStream).toHaveBeenCalledTimes(1);
      expect(retryModel.doStream).toHaveBeenCalledTimes(0);
      expect(error).toBeUndefined();
      expect(chunksToText(chunks)).toBe(mockResultText);
    });

    it('should retry for status 529', async () => {
      // Arrange
      const baseModel = new MockLanguageModel({ doStream: overloadedError });
      const retryModel = new MockLanguageModel({
        doStream: {
          stream: convertArrayToReadableStream(mockStreamChunks),
        },
      });
      let error: unknown;

      // Act
      const result = streamText({
        model: createRetryable({
          model: baseModel,
          retries: [serviceOverloaded(retryModel)],
        }),
        prompt: 'Hello!',
        onError(data) {
          error = data.error;
        },
      });

      const chunks = await convertAsyncIterableToArray(result.fullStream);

      // Assert
      expect(baseModel.doStream).toHaveBeenCalledTimes(1);
      expect(retryModel.doStream).toHaveBeenCalledTimes(1);
      expect(error).toBeUndefined();
      expect(chunksToText(chunks)).toBe(mockResultText);
    });

    it('should not retry if no matches', async () => {
      // Arrange
      const baseModel = new MockLanguageModel({
        doStream: new APICallError({
          message: 'Some other error',
          url: '',
          requestBodyValues: {},
          statusCode: 400,
          responseHeaders: {},
          responseBody: '{}',
          isRetryable: false,
          data: {
            error: {
              message: 'Some other error',
              type: null,
              param: 'prompt',
              code: 'other_error',
            },
          },
        }),
      });

      const retryModel = new MockLanguageModel({
        doStream: {
          stream: convertArrayToReadableStream(mockStreamChunks),
        },
      });
      let error: unknown;

      // Act
      const result = streamText({
        model: createRetryable({
          model: baseModel,
          retries: [serviceOverloaded(retryModel)],
        }),
        prompt: 'Hello!',
        maxRetries: 0,
        onError(data) {
          error = data.error;
        },
      });

      const chunks = await convertAsyncIterableToArray(result.fullStream);

      // Assert
      expect(baseModel.doStream).toHaveBeenCalledTimes(1);
      expect(retryModel.doStream).toHaveBeenCalledTimes(0);
      expect(error).toBeDefined();
      expect(chunks).toMatchInlineSnapshot(`
        [
          {
            "type": "start",
          },
          {
            "error": [AI_APICallError: Some other error],
            "type": "error",
          },
        ]
      `);
    });

    // overloaded error has been integrated into AI SDK, so we don't need to test it here
    describe.skip('anthropic', () => {
      process.env.ANTHROPIC_API_KEY = 'test-anthropic-key';
      process.env.OPENAI_API_KEY = 'test-openai-key';

      const server = createTestServer({
        'https://api.anthropic.com/v1/messages': {},
        'https://api.openai.com/v1/responses': {
          response: {
            type: 'stream-chunks',
            chunks: [...textChunks.openai(['Hello', ', ', 'World!'])],
          },
        },
      });

      it('should retry when overloaded error occurs at stream creation', async () => {
        // Arrange
        server.urls['https://api.anthropic.com/v1/messages'].response = {
          type: 'error',
          status: 529,
          body: '{"type":"error","error":{"details":null,"type":"overloaded_error","message":"Overloaded"}}',
        };

        const baseModel = anthropic('claude-sonnet-4-20250514');
        const retryModel = openai('gpt-5-2025-08-07');

        const baseModelSpy = vi.spyOn(baseModel, 'doStream');
        const retryModelSpy = vi.spyOn(retryModel, 'doStream');

        // Act
        const result = streamText({
          model: createRetryable({
            model: baseModel,
            retries: [serviceOverloaded(retryModel)],
          }),
          prompt: 'Hello!',
          onError(err) {
            // Errors are automatically retried, so this should not be called
            expect.unreachable('Should not log any errors');
          },
        });

        const chunks = await convertAsyncIterableToArray(result.textStream);

        // Assert
        expect(baseModelSpy).toHaveBeenCalledTimes(1);
        expect(retryModelSpy).toHaveBeenCalledTimes(1);
        expect(chunks).toMatchInlineSnapshot(`
          [
            "Hello",
            ", ",
            "World!",
          ]
        `);
      });

      it('should retry when overloaded error occurs at the stream start', async () => {
        // Arrange
        server.urls['https://api.anthropic.com/v1/messages'].response = {
          type: 'stream-chunks',
          chunks: [
            `data: {"type":"error","error":{"details":null,"type":"overloaded_error","message":"Overloaded"}}\n\n`,
          ],
        };

        const baseModel = anthropic('claude-sonnet-4-20250514');
        const retryModel = openai('gpt-5-2025-08-07');

        const baseModelSpy = vi.spyOn(baseModel, 'doStream');
        const retryModelSpy = vi.spyOn(retryModel, 'doStream');

        // Act
        const result = streamText({
          model: createRetryable({
            model: baseModel,
            retries: [serviceOverloaded(retryModel)],
          }),
          prompt: 'Hello!',
          onError(err) {
            // Errors are automatically retried, so this should not be called
            expect.unreachable('Should not log any errors');
          },
        });

        const chunks = await convertAsyncIterableToArray(result.textStream);

        // Assert
        expect(baseModelSpy).toHaveBeenCalledTimes(1);
        expect(retryModelSpy).toHaveBeenCalledTimes(1);
        expect(chunks).toMatchInlineSnapshot(`
          [
            "Hello",
            ", ",
            "World!",
          ]
        `);
      });

      it('should NOT retry when overloaded error occurs during streaming', async () => {
        // Arrange
        server.urls['https://api.anthropic.com/v1/messages'].response = {
          type: 'stream-chunks',
          chunks: [
            `data: {"type":"message_start","message":{"id":"msg_01KfpJoAEabmH2iHRRFjQMAG","type":"message","role":"assistant","content":[],"model":"claude-3-haiku-20240307","stop_reason":null,"stop_sequence":null,"usage":{"input_tokens":17,"output_tokens":1}}}\n\n`,
            `data: {"type":"content_block_start","index":0,"content_block":{"type":"text","text":""}}\n\n`,
            `data: {"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"Hello"}}\n\n`,
            `data: {"type":"error","error":{"details":null,"type":"overloaded_error","message":"Overloaded"}}\n\n`,
          ],
        };

        const baseModel = anthropic('claude-sonnet-4-20250514');
        const retryModel = openai('gpt-5-2025-08-07');

        const baseModelSpy = vi.spyOn(baseModel, 'doStream');
        const retryModelSpy = vi.spyOn(retryModel, 'doStream');

        // Act
        const result = streamText({
          model: createRetryable({
            model: baseModel,
            retries: [serviceOverloaded(retryModel)],
          }),
          prompt: 'Hello!',
        });

        const chunks = await convertAsyncIterableToArray(result.textStream);

        // Assert
        expect(baseModelSpy).toHaveBeenCalledTimes(1);
        expect(retryModelSpy).toHaveBeenCalledTimes(0);
        expect(chunks).toMatchInlineSnapshot(`
          [
            "Hello",
          ]
        `);
      });
    });
  });

  describe('embed', () => {
    it('should succeed without errors', async () => {
      // Arrange
      const baseModel = new MockEmbeddingModel({ doEmbed: mockEmbeddings });
      const retryModel = new MockEmbeddingModel({ doEmbed: mockEmbeddings });

      // Act
      const result = await embed({
        model: createRetryable({
          model: baseModel,
          retries: [serviceOverloaded(retryModel)],
        }),
        value: 'Hello!',
      });

      // Assert
      expect(baseModel.doEmbed).toHaveBeenCalledTimes(1);
      expect(result.embedding).toBe(mockEmbeddings.embeddings[0]);
    });

    it('should retry for status 529', async () => {
      // Arrange
      const baseModel = new MockEmbeddingModel({ doEmbed: overloadedError });
      const retryModel = new MockEmbeddingModel({ doEmbed: mockEmbeddings });

      // Act
      const result = await embed({
        model: createRetryable({
          model: baseModel,
          retries: [serviceOverloaded(retryModel)],
        }),
        value: 'Hello!',
      });

      // Assert
      expect(baseModel.doEmbed).toHaveBeenCalledTimes(1);
      expect(retryModel.doEmbed).toHaveBeenCalledTimes(1);
      expect(result.embedding).toBe(mockEmbeddings.embeddings[0]);
    });

    it('should not retry for status 200', async () => {
      // Arrange
      const baseModel = new MockEmbeddingModel({ doEmbed: mockEmbeddings });
      const retryModel = new MockEmbeddingModel({ doEmbed: mockEmbeddings });

      // Act
      const result = await embed({
        model: createRetryable({
          model: baseModel,
          retries: [serviceOverloaded(retryModel)],
        }),
        value: 'Hello!',
      });

      // Assert
      expect(baseModel.doEmbed).toHaveBeenCalledTimes(1);
      expect(retryModel.doEmbed).toHaveBeenCalledTimes(0);
      expect(result.embedding).toBe(mockEmbeddings.embeddings[0]);
    });

    it('should not retry if no matches', async () => {
      // Arrange
      const baseModel = new MockEmbeddingModel({
        doEmbed: new APICallError({
          message: 'Some other error',
          url: '',
          requestBodyValues: {},
          statusCode: 400,
          responseHeaders: {},
          responseBody: '{}',
          isRetryable: false,
          data: {
            error: {
              message: 'Some other error',
              type: null,
              param: 'prompt',
              code: 'other_error',
            },
          },
        }),
      });

      const retryModel = new MockEmbeddingModel({ doEmbed: mockEmbeddings });

      // Act
      const result = embed({
        model: createRetryable({
          model: baseModel,
          retries: [serviceOverloaded(retryModel)],
        }),
        value: 'Hello!',
        maxRetries: 0,
      });

      // Assert
      await expect(result).rejects.toThrowError(APICallError);
      expect(baseModel.doEmbed).toHaveBeenCalledTimes(1);
      expect(retryModel.doEmbed).toHaveBeenCalledTimes(0);
    });
  });
});
