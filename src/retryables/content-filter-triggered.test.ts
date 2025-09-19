import { describe } from 'node:test';
import type { LanguageModelV2 } from '@ai-sdk/provider';
import { APICallError, generateText, NoObjectGeneratedError } from 'ai';
import { expect, it } from 'vitest';
import { createRetryable } from '../create-retryable-model.js';
import { createMockModel } from '../test-utils.js';
import { contentFilterTriggered } from './content-filter-triggered.js';

type LanguageModelV2GenerateFn = LanguageModelV2['doGenerate'];

type LanguageModelV2GenerateResult = Awaited<
  ReturnType<LanguageModelV2GenerateFn>
>;

const mockResultText = 'Hello, world!';

const mockResult: LanguageModelV2GenerateResult = {
  finishReason: 'stop',
  usage: { inputTokens: 10, outputTokens: 20, totalTokens: 30 },
  content: [{ type: 'text', text: mockResultText }],
  warnings: [],
};

const errors = [
  new APICallError({
    message:
      "The response was filtered due to the prompt triggering Azure OpenAI's content management policy. Please modify your prompt and retry. To learn more about our content filtering policies please read our documentation: https://go.microsoft.com/fwlink/?linkid=2198766",
    url: '',
    requestBodyValues: {},
    statusCode: 400,
    responseHeaders: {},
    responseBody:
      '{"error":{"message":"The response was filtered due to the prompt triggering Azure OpenAI\'s content management policy. Please modify your prompt and retry. To learn more about our content filtering policies please read our documentation: https://go.microsoft.com/fwlink/?linkid=2198766","type":null,"param":"prompt","code":"content_filter","status":400,"innererror":{"code":"ResponsibleAIPolicyViolation","content_filter_result":{"hate":{"filtered":false,"severity":"safe"},"self_harm":{"filtered":false,"severity":"safe"},"sexual":{"filtered":true,"severity":"high"},"violence":{"filtered":false,"severity":"safe"}}}}}',
    isRetryable: false,
    data: {
      error: {
        message:
          "The response was filtered due to the prompt triggering Azure OpenAI's content management policy. Please modify your prompt and retry. To learn more about our content filtering policies please read our documentation: https://go.microsoft.com/fwlink/?linkid=2198766",
        type: null,
        param: 'prompt',
        code: 'content_filter',
      },
    },
  }),
  new NoObjectGeneratedError({
    response: {
      id: 'mock-response-id',
      modelId: 'mock-model-id',
      timestamp: new Date(),
    },
    finishReason: 'content-filter',
    usage: { inputTokens: 10, outputTokens: 20, totalTokens: 30 },
  }),
];

describe('contentFilterTriggered', () => {
  it('should succeed without errors', async () => {
    // Arrange
    const baseModel = createMockModel(mockResult);
    const retryModel = createMockModel(mockResult);

    // Act
    const result = await generateText({
      model: createRetryable({
        model: baseModel,
        retries: [contentFilterTriggered(retryModel)],
      }),
      prompt: 'Hello!',
    });

    // Assert
    expect(baseModel.doGenerateCalls.length).toBe(1);
    expect(result.text).toBe(mockResultText);
  });

  it.each(errors)(
    'should retry in case of content filter error: $name',
    async (error) => {
      // Arrange
      const baseModel = createMockModel(error);
      const retryModel = createMockModel(mockResult);

      // Act
      const result = await generateText({
        model: createRetryable({
          model: baseModel,
          retries: [contentFilterTriggered(retryModel)],
        }),
        prompt: 'Hello!',
        maxRetries: 0,
      });

      // Assert
      expect(baseModel.doGenerateCalls.length).toBe(1);
      expect(retryModel.doGenerateCalls.length).toBe(1);
      expect(result.text).toBe(mockResultText);
    },
  );

  it('should not retry if no matches', async () => {
    // Arrange
    const baseModel = createMockModel(
      new APICallError({
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
    );

    const retryModel = createMockModel(mockResult);

    // Act
    const result = generateText({
      model: createRetryable({
        model: baseModel,
        retries: [contentFilterTriggered(retryModel)],
      }),
      prompt: 'Hello!',
      maxRetries: 0,
    });

    // Assert
    await expect(result).rejects.toThrowError(APICallError);
    expect(baseModel.doGenerateCalls.length).toBe(1);
    expect(retryModel.doGenerateCalls.length).toBe(0);
  });
});
