import { readFileSync, writeFileSync } from 'node:fs';

const file = 'src/create-retryable-model.test.ts';
let content = readFileSync(file, 'utf-8');

// Fix contentFilterResult references - embeddings don't have finishReason, so we need to remove those tests
content = content.replace(
  /doEmbed: contentFilterResult,?/g,
  'doEmbed: mockEmbeddings,',
);

// Remove result-based retry tests from embed block as they don't apply to embeddings
const lines = content.split('\n');
const startMarker = "describe('embed', () => {";
const endMarker = '});';

let inEmbedBlock = false;
let inResultBasedBlock = false;
let blockDepth = 0;
const newLines: string[] = [];

for (let i = 0; i < lines.length; i++) {
  const line = lines[i];

  if (line.includes(startMarker)) {
    inEmbedBlock = true;
  }

  if (inEmbedBlock && line.includes("describe('result-based retries'")) {
    inResultBasedBlock = true;
    blockDepth = 1;
    continue; // Skip this line
  }

  if (inResultBasedBlock) {
    // Count braces to track depth
    for (const char of line) {
      if (char === '{') blockDepth++;
      if (char === '}') blockDepth--;
    }

    // When we exit the result-based block
    if (blockDepth === 0) {
      inResultBasedBlock = false;
    }
    continue; // Skip lines in result-based block
  }

  // Fix 'should NOT call onError handler for result-based retries' test
  if (
    inEmbedBlock &&
    line.includes('should NOT call onError handler for result-based retries')
  ) {
    // Skip until the end of this test
    let testDepth = 1;
    i++; // Skip current line
    while (i < lines.length && testDepth > 0) {
      const testLine = lines[i];
      for (const char of testLine) {
        if (char === '{') testDepth++;
        if (char === '}') testDepth--;
      }
      i++;
    }
    i--; // Back up one line
    continue;
  }

  // Fix 'should call onRetry handler for result-based retries' test
  if (
    inEmbedBlock &&
    line.includes('should call onRetry handler for result-based retries')
  ) {
    // Skip until the end of this test
    let testDepth = 1;
    i++; // Skip current line
    while (i < lines.length && testDepth > 0) {
      const testLine = lines[i];
      for (const char of testLine) {
        if (char === '{') testDepth++;
        if (char === '}') testDepth--;
      }
      i++;
    }
    i--; // Back up one line
    continue;
  }

  // Fix 'ignore plain language models' to 'ignore plain embedding models'
  if (inEmbedBlock && line.includes('should ignore plain language models')) {
    newLines.push(
      line.replace(
        'should ignore plain language models',
        'should ignore plain embedding models',
      ),
    );
    continue;
  }

  newLines.push(line);

  if (inEmbedBlock && line === '});' && !line.includes('describe')) {
    // Check if this closes the embed block
    const nextLine = lines[i + 1];
    if (
      nextLine &&
      !nextLine.trim().startsWith('describe') &&
      !nextLine.trim().startsWith('it')
    ) {
      inEmbedBlock = false;
    }
  }
}

content = newLines.join('\n');

writeFileSync(file, content, 'utf-8');

console.log('✅ Fixed embed block types in create-retryable-model.test.ts');
