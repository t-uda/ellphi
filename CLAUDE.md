# Claude Code Guidelines

This file is loaded automatically by Claude Code at session start.
For general project conventions and CI requirements, read `AGENTS.md` first.

## Subagent Permissions

When orchestrating work through Claude Code subagents, the following
permissions must be present in `.claude/settings.local.json`:

```json
{
  "permissions": {
    "allow": [
      "Edit",
      "Write",
      "Bash(git:*)",
      "Bash(cp:*)",
      "Bash(python3:*)"
    ]
  }
}
```

Notes:
- Permission changes take effect after a session restart.
- Binary file processing (e.g. image transparency) should be handled by
  the main agent, not delegated to subagents.
- Simple one-off shell commands that fall outside the patterns above
  can be run by the main agent directly.

## Orchestration Style

- Delegate file editing and git commits to background subagents.
- Use the main agent for discussion, planning, and binary/image processing.
- Commit granularity: one logical change per commit; a subagent handling
  multiple related files in one pass may bundle them into a single commit.
