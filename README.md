# TypeScript Single Package Project Template

This template provides an opinionated setup for a single package TypeScript project.

## 🚀 Features

- 📦 [PNPM](https://pnpm.io/) for efficient package management
- 🧹 [Biome](https://biomejs.dev/) for linting and formatting
- 🧪 [Vitest](https://vitest.dev/) for fast, modern testing
- 🏗️ [unbuild](https://github.com/unjs/unbuild) for TypeScript building and bundling
- 🏃‍♂️ [tsx](https://tsx.is/) for running TypeScript files
- 🐶 [Husky](https://github.com/typicode/husky) for Git hooks
- 🔄 [GitHub Actions](.github/workflows/ci.yml) for continuous integration
- 🐞 [VSCode](.vscode/) debug configuration and editor settings
- 🔧 [@total-typescript/tsconfig](https://github.com/total-typescript/tsconfig) for TypeScript configuration
- 🎯 [Are The Types Wrong?](https://github.com/arethetypeswrong/arethetypeswrong.github.io) for type validation

## 🚀 Getting Started

### 1. Create a new repository

Create a new repository [using this template](https://docs.github.com/en/repositories/creating-and-managing-repositories/creating-a-repository-from-a-template)

### 2. Replace placeholders

Replace all occurences of the following placeholders with the correct values:

| Placeholder | File | Description |
| --- | --- | --- |
| `<PACKAGE>` | `package.json` | Your package name |
| `<DESCRIPTION>` | `package.json` | Your package description |
| `<USERNAME>` | `package.json` | Your GitHub username |
| `<REPO>` | `package.json` | Your repository name |
| `<AUTHOR>` | `package.json` | Your name |
| `<LICENSE>` | `package.json` | Your license |

### 3. Apply ToDos

Find all occurrences of `TODO` and apply them:

| TODO | File | Description |
| --- | --- | --- |
| `TODO: PREVIEW` | `.github/workflows/ci.yml` | Create [preview releases](#preview-releases) |
| `TODO: PUBLISH` | `.github/workflows/ci.yml` | [Publish to NPM](#publish-npm) |

### 4. Install, Build, Test

Verify your project is working by running `install`, `build`, and `test`:

```sh
pnpm install
pnpm build
pnpm test
```

Happy coding! 🎉

## 📋 Details

### Package

The [`package.json`](package.json) is configured as ESM (`"type": "module"`), but supports dual publishing with both ESM and CJS module formats.

### Biome

[`biome.jsonc`](biome.jsonc) contains the default [Biome configuration](https://biomejs.dev/reference/configuration/) with minimal formatting adjustments. It uses the formatter settings from the [`.editorconfig`](.editorconfig) file.

### Vitest

An empty Vitest config is provided in [`vitest.config.ts`](vitest.config.ts).

### Build and Run

- `unbuild` builds `./src/index.ts`, outputting both ESM and CJS formats to the `dist` folder.
- `tsx` compiles and runs TypeScript files on-the-fly.

### Git Hooks

[Husky](https://github.com/typicode/husky) runs the [.husky/pre-commit](.husky/pre-commit) hook to lint staged files.

### Continuous Integration

[`.github/workflows/ci.yml`](.github/workflows/ci.yml) defines a GitHub Actions workflow to run linting and tests on commits and pull requests.

### VSCode Integration

#### Debugging

[`.vscode/launch.json`](.vscode/launch.json) provides VSCode launch configurations:
- `Debug (tsx)`: Run and debug TypeScript files
- `Test (vitest)`: Debug tests

It uses the [JavaScript Debug Terminal](https://code.visualstudio.com/docs/nodejs/nodejs-debugging) to run and debug.

#### Editor Settings

[`.vscode/settings.json`](.vscode/settings.json) configures Biome as the formatter and enables format-on-save.

### EditorConfig

[`.editorconfig`](.editorconfig) ensures consistent coding styles across different editors and IDEs:

- Uses spaces for indentation (2 spaces)
- Sets UTF-8 charset
- Ensures LF line endings
- Trims trailing whitespace (except in Markdown files)
- Inserts a final newline in files

This configuration complements Biome and helps maintain a consistent code style throughout the project.

### Type Validation

The project includes the `@arethetypeswrong/cli` CLI tool to validate TypeScript types in your package. Run `pnpm typecheck` after building to ensure your package's types are correct and compatible with both ESM and CommonJS environments.

## Optional

### <a name="publish-npm"></a> Publish to NPM
[JS-DevTools/npm-publish](https://github.com/JS-DevTools/npm-publish) is a GitHub Action to publish packages to npm automatically by updating the version number.

To enable this, apply the `TODO: PUBLISH`.

### <a name="preview-releases"></a> Preview Releases

[pkg.pr.new](https://github.com/stackblitz-labs/pkg.pr.new) will automatically generate preview releases for every push and pull request. This allows you to test changes before publishing to npm.

Must install GitHub App: [pkg.pr.new](https://github.com/apps/pkg-pr-new)

To enable this, apply the `TODO: PREVIEW`.
