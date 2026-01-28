---
description: Suggest which slash commands could be useful right now
---

Analyze the current conversation context and suggest which slash commands would be most useful to run next.

## Available Slash Commands

Review each command and determine if it's relevant to the current situation:

### /lint
**Suggest when:**
- User just finished writing or editing code
- Code has been modified but not formatted
- Before committing changes
- User mentions code style or formatting issues

### /refactor
**Suggest when:**
- User mentions code is messy, hard to read, or needs cleanup
- Functions or files have grown large
- User asks about code quality or structure
- After implementing a feature (to clean up)

### /add-tests
**Suggest when:**
- User just implemented new functionality
- User mentions testing or test coverage
- Before marking a feature as complete
- User asks about code reliability

### /add-print-diagnostics
**Suggest when:**
- User is debugging an issue
- Training results are unexpected
- User wants to understand what's happening during execution
- Something isn't working and cause is unclear

### /homework-formatting
**Suggest when:**
- User mentions DS2500 or homework
- Code needs PEP 8 spacing for submission
- User is preparing code for grading

## Output Format

Based on your analysis, provide suggestions in this format:

```
## Suggested Commands

### Highly Relevant
- `/command` - Why it's useful right now

### Possibly Useful
- `/command` - Why it might help

### Not Applicable
- Commands that don't fit the current context (brief explanation)
```

Be concise and specific about WHY each command is or isn't relevant to the current conversation.
