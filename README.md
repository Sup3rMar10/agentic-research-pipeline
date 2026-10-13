# Agentic Research Pipeline

A small multi-agent pipeline built on the [Anthropic Python SDK](https://github.com/anthropics/anthropic-sdk-python). Give it a topic and it returns a cited one-page brief.

```
topic ──> Planner ──> Researcher (tool loop, per question) ──> Writer <──> Critic ──> brief
```

| Agent | Job | Pattern |
|---|---|---|
| **Planner** | Splits the topic into 3–4 research questions | Single call, low effort |
| **Researcher** | Fetches sources and reports findings | Tool-using agent via the SDK tool runner |
| **Writer** | Drafts the brief from the notes | Single call |
| **Critic** | Fact-checks the draft, then approves it or sends it back | Review loop, capped at 2 revisions |

## Run

```bash
pip install -r requirements.txt
export ANTHROPIC_API_KEY=...
python pipeline.py "How do vector databases work?"
```

## Design notes

- The whole pipeline is one file (~100 lines) so the control flow is easy to read.
- The Researcher uses `@beta_tool` and `client.beta.messages.tool_runner`, so the SDK runs the tool loop.
- Each agent is a role plus a prompt, and reasoning effort is tuned per role.
- The revision loop is capped so the pipeline always terminates.
