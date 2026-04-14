---
name: commit
description: Create a git commit with an auto-generated conventional commit message summarizing the changes. Optionally push to remote.
---

# Create Commit

Create a commit with an auto-generated message summarizing the relevant changes for the current task.

## Repository Context

- Working tree status: !`git status`
- Unstaged changes: !`git diff`
- Staged changes: !`git diff --cached`
- Recent commits (style reference): !`git log -5 --format='%h %s'`

## Step 1: Check for Changes

If there are no changes in the context above (nothing staged, unstaged, or untracked): stop and tell the user there's nothing to commit.

## Step 2: Read Untracked Files

For untracked files shown in the status above, read them with the Read tool to understand their contents. Diffs already cover modified files.

## Step 3: Filter to Relevant Changes

Evaluate each changed/untracked file against the apparent intent of the current task. Classify each as **relevant** or **unrelated**.

Mark a file as unrelated if it:
- Belongs to a different feature or fix unrelated to the current task
- Is an auto-generated artifact (lockfiles, build output, `.cache/`) not caused by this task
- Appears accidentally modified (e.g. whitespace-only changes in unrelated files)

If unsure about any file, ask the user before proceeding:
> "These files are also changed — are they part of this commit?
> - `path/to/file1`
> - `path/to/file2`"

Only carry the relevant files forward into Step 4.

## Step 4: Draft Commit Message

Write a conventional commit message:
- **Type:** `feat:`, `fix:`, `chore:`, `refactor:`, `test:`, `docs:`, etc.
- **Scope (optional):** package name if changes are scoped to one package, e.g. `feat(api):`
- **Subject:** concise summary of what changed and why (under 72 chars)
- **Body (if needed):** additional context for complex changes, separated by a blank line

If multiple unrelated changes exist, suggest the user split them into separate commits.

## Step 5: Stage and Commit

1. Stage only the relevant files identified in Step 3 with `git add <specific files>` — never use `git add -A`
2. Create the commit using a HEREDOC:

```
git commit -m "$(cat <<'EOF'
<type>(<scope>): <subject>

<optional body>
EOF
)"
```

3. Run `git status` to verify the commit succeeded.

## Step 6: Ask About Push

After a successful commit, ask the user: **"Push to remote?"**

- If yes: check for upstream with `git rev-parse --abbrev-ref @{upstream} 2>/dev/null`, then `git push -u origin HEAD` (no upstream) or `git push` (has upstream)
- If no: done
