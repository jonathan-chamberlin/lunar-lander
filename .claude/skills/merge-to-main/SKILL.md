---
name: merge-to-main
description: Merge the current branch into the main branch and push to origin. Use when user asks to merge changes into main, put changes into main branch, or merge current branch to main.
allowed-tools: Bash
---

# Merge to Main: Merge Current Branch into Main

Merge the current feature branch into the main branch and push to origin. Handles worktrees automatically.

## When to Use This Skill

- User asks to "merge changes into main"
- User wants to "merge to main" or "merge into main"
- User says "put changes into main branch"
- User wants to "merge current branch to main"

## Merge Workflow

Execute these exact steps in order:

### Step 1: Check if currently in a worktree

```bash
git worktree list
```

This shows all worktrees. Example output:
```
C:/Repositories for Git/lunar-lander-file-folder/lunar-lander                 abc1234 [main]
C:/Repositories for Git/lunar-lander-file-folder/lunar-lander-experiment      def5678 [experiment-branch]
```

Note:
- If there's only one entry, you're in the main working tree
- If there are multiple entries, identify which worktree you're in and which branch it's on
- **Save the branch name you want to merge** (e.g., `experiment-branch`)
- **Save the worktree path if you're in one** (e.g., `lunar-lander-experiment`)

### Step 2: Check current branch

```bash
git branch
```

Note the current branch (marked with `*`). This is the branch you'll merge into main.

### Step 3: Navigate to the main repository

```bash
cd "C:\Repositories for Git\lunar-lander-file-folder\lunar-lander"
```

**Note:** Always go to the main `lunar-lander` directory, not a worktree folder.

### Step 4: Switch to the main branch

```bash
git checkout main
```

### Step 5: Merge the feature branch into main

```bash
git merge <branch-name>
```

Replace `<branch-name>` with the branch you want to merge (e.g., `experiment-branch`).

### Step 6: Push the updated main branch to origin

```bash
git push origin main
```

### Step 7: Remove the worktree (if one was used)

If the branch was in a worktree, remove it now that the merge is complete:

```bash
git worktree remove ../lunar-lander-<branch-name>
```

**Example:** If the worktree folder was `lunar-lander-experiment`:
```bash
git worktree remove ../lunar-lander-experiment
```

### Step 8: Delete the merged branch (optional)

After merging, ask the user if they want to delete the feature branch. If yes, delete both the local and remote branches:

```bash
# Delete local branch
git branch -d <branch-name>

# Delete remote branch from GitHub
git push origin --delete <branch-name>
```

**Important:** Always delete both local AND remote branches, otherwise the branch will still appear on GitHub. Even if it doesn't seem like the branch was put on github, still check if it's there and if so delete it.

**WARNING: NEVER delete `main` or `origin/main`. Only delete feature branches.**

## Important Notes

- **NEVER delete `main` or `origin/main`** - only delete feature branches after merging
- Always verify you're in the correct directory (`lunar-lander` subdirectory) before running git commands
- Use `git branch` to confirm which branch you're on before merging
- If there are merge conflicts, resolve them before pushing
- Close any VS Code windows open to the worktree folder before removing it
- The worktree folder will be deleted when you run `git worktree remove`
- If you get an error when attempting to delete it, it's possible the user still has the worktree open, in which case ask if they have closed it before you try again. 
