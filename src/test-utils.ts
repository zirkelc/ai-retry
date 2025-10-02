import type { LanguageModelV2 } from '@ai-sdk/provider';
import type { generateText, streamText, TextStreamPart } from 'ai';
import { MockLanguageModelV2 } from 'ai/test';

type StreamText = Parameters<typeof streamText>[0];
type GenerateText = Parameters<typeof generateText>[0];

export type LanguageModelV2GenerateFn = LanguageModelV2['doGenerate'];
export type LanguageModelV2StreamFn = LanguageModelV2['doStream'];

export type LanguageModelV2GenerateResult = Awaited<
  ReturnType<LanguageModelV2GenerateFn>
>;

export type LanguageModelV2StreamResult = Awaited<
  ReturnType<LanguageModelV2StreamFn>
>;

const mockGenerateId = () => 'aitxt-mock-id';
const mockCurrentDate = () => new Date(0);

export const mockGenerateOptions: Partial<GenerateText> = {
  _internal: {
    generateId: mockGenerateId,
    currentDate: mockCurrentDate,
  },
};

export const mockStreamOptions: Pick<StreamText, '_internal'> = {
  _internal: {
    generateId: mockGenerateId,
    currentDate: mockCurrentDate,
  },
};

let mockModelCounter = 0;
const generateMockModelId = () => {
  mockModelCounter += 1;
  return currentMockModelId();
};

const provider = 'mock-provider';
export const currentMockModelId = () => `mock-model-${mockModelCounter}`;

export const createMockModel = (
  resultOrFunction:
    | LanguageModelV2GenerateResult
    | LanguageModelV2GenerateFn
    | Error,
) => {
  const modelId = generateMockModelId();

  if (resultOrFunction instanceof Error)
    return new MockLanguageModelV2({
      provider,
      modelId,
      doGenerate: async () => {
        throw resultOrFunction;
      },
    });

  if (typeof resultOrFunction === 'function')
    return new MockLanguageModelV2({
      provider,
      modelId,
      doGenerate: resultOrFunction,
    });

  return new MockLanguageModelV2({
    provider,
    modelId,
    doGenerate: async () => resultOrFunction,
  });
};

export const createMockStreamingModel = (
  resultOrFunction:
    | LanguageModelV2StreamResult
    | LanguageModelV2StreamFn
    | Error,
) => {
  const modelId = generateMockModelId();

  if (resultOrFunction instanceof Error)
    return new MockLanguageModelV2({
      provider,
      modelId,
      doStream: async () => {
        throw resultOrFunction;
      },
    });

  if (typeof resultOrFunction === 'function')
    return new MockLanguageModelV2({
      provider,
      modelId,
      doStream: resultOrFunction,
    });

  return new MockLanguageModelV2({
    provider,
    modelId,
    doStream: async () => resultOrFunction,
  });
};

export const chunksToText = (chunks: TextStreamPart<any>[]): string => {
  return chunks
    .map((chunk) => (chunk.type === 'text-delta' ? chunk.text : ''))
    .join('');
};

export const errorFromChunks = (
  chunks: TextStreamPart<any>[],
): unknown | null => {
  const errorChunk = chunks.find((chunk) => chunk.type === 'error');
  return errorChunk && errorChunk.type === 'error' ? errorChunk.error : null;
};

export const errorChunk = (error: Record<string, any>): string =>
  `data: {"type":"error","error":${JSON.stringify(error)}}\n\n`;

export const textChunks = {
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
