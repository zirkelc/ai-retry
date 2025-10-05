import { readFileSync, writeFileSync } from 'node:fs';

const file = 'src/create-retryable-model.test.ts';
let content = readFileSync(file, 'utf-8');

// Get the lines
const lines = content.split('\n');

// Find the embed block (starts at line 1616, 0-indexed)
const startLine = 1616; // 0-indexed line 1617
const endLine =
  lines.findIndex((line, idx) => idx > startLine && line === '});') + 1;

console.log(`Found embed block from line ${startLine + 1} to ${endLine + 1}`);

// Replace within the embed block only
for (let i = startLine; i < endLine; i++) {
  // Replace MockLanguageModel with MockEmbeddingModel
  lines[i] = lines[i].replace(/MockLanguageModel/g, 'MockEmbeddingModel');

  // Replace doGenerate with doEmbed
  lines[i] = lines[i].replace(/doGenerate/g, 'doEmbed');

  // Replace generateText with embed
  lines[i] = lines[i].replace(/generateText/g, 'embed');

  // Replace prompt: 'Hello!' with value: 'Hello!'
  lines[i] = lines[i].replace(/prompt: 'Hello!'/g, "value: 'Hello!'");

  // Replace mockResult with mockEmbeddings
  lines[i] = lines[i].replace(/: mockResult/g, ': mockEmbeddings');

  // Replace result.text with result.embedding
  lines[i] = lines[i].replace(/result\.text/g, 'result.embedding');

  // Replace mockResultText references
  lines[i] = lines[i].replace(
    /toBe\(mockResultText\)/g,
    'toEqual(mockEmbeddings.embeddings[0])',
  );

  // Replace Retryable<LanguageModelV2> with just Retryable
  lines[i] = lines[i].replace(/Retryable<LanguageModelV2>/g, 'Retryable');

  // Fix comments
  lines[i] = lines[i].replace(
    /should generate text successfully/g,
    'should embed successfully',
  );
  lines[i] = lines[i].replace(
    /should use plain language models/g,
    'should use plain embedding models',
  );
}

// Write back
content = lines.join('\n');
writeFileSync(file, content, 'utf-8');

console.log('✅ Refactored embed block in create-retryable-model.test.ts');
