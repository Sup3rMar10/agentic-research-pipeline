"""Planner -> Researcher -> Writer <-> Critic: a small multi-agent pipeline on the Anthropic SDK.

Usage: python pipeline.py "How do vector databases work?"
"""
import sys
import urllib.request

import anthropic
from anthropic import beta_tool

MODEL = "claude-opus-5-5"
MAX_REVISIONS = 2

client = anthropic.Anthropic()  # reads ANTHROPIC_API_KEY from the environment


def text_of(message) -> str:
    if message.stop_reason == "refusal":
        raise RuntimeError(f"Model refused: {message.stop_details}")
    return "".join(b.text for b in message.content if b.type == "text").strip()


def ask(system: str, prompt: str, effort: str = "medium") -> str:
    """Single-shot agent: one role, one prompt, no tools."""
    message = client.messages.create(
        model=MODEL,
        max_tokens=16000,
        system=system,
        thinking={"type": "adaptive"},
        output_config={"effort": effort},
        messages=[{"role": "user", "content": prompt}],
    )
    return text_of(message)


@beta_tool
def fetch_url(url: str) -> str:
    """Fetch a web page and return its first 8,000 characters of text.

    Args:
        url: Absolute http(s) URL to fetch.
    """
    req = urllib.request.Request(url, headers={"User-Agent": "research-pipeline/0.1"})
    with urllib.request.urlopen(req, timeout=15) as resp:
        return resp.read(8000).decode("utf-8", errors="replace")


def plan(topic: str) -> list[str]:
    """Planner agent: break the topic into focused research questions."""
    out = ask(
        "You are a research planner. Reply with 3 to 4 focused research questions, one per line, no numbering.",
        topic,
        effort="low",
    )
    return [line.strip() for line in out.splitlines() if line.strip()]


def research(question: str) -> str:
    """Researcher agent: tool-using loop (the SDK tool runner drives it) that gathers findings."""
    runner = client.beta.messages.tool_runner(
        model=MODEL,
        max_tokens=16000,
        system="You are a researcher. Use fetch_url on reputable sources, then report concise findings with source URLs.",
        tools=[fetch_url],
        thinking={"type": "adaptive"},
        messages=[{"role": "user", "content": question}],
    )
    final = None
    for message in runner:  # one iteration per model turn; the runner executes tool calls
        final = message
    return text_of(final)


def write(topic: str, notes: str, feedback: str = "", draft: str = "") -> str:
    """Writer agent: turn notes into a brief, optionally revising against critic feedback."""
    prompt = f"Topic: {topic}\n\nResearch notes:\n{notes}"
    if feedback:
        prompt += f"\n\nPrevious draft:\n{draft}\n\nCritic feedback to address:\n{feedback}"
    return ask("You are a technical writer. Produce a clear, well-structured one-page brief in Markdown with cited sources.", prompt)


def critique(brief: str, notes: str) -> str:
    """Critic agent: approve, or return concrete feedback."""
    return ask(
        "You are a strict fact-checking editor. Check the brief against the notes. "
        "Reply exactly 'APPROVED' if it is accurate and clear, otherwise list specific fixes.",
        f"Notes:\n{notes}\n\nBrief:\n{brief}",
        effort="high",
    )


def run(topic: str) -> str:
    questions = plan(topic)
    print(f"[planner] {len(questions)} questions")
    notes = "\n\n".join(f"## {q}\n{research(q)}" for q in questions)
    print("[researcher] notes gathered")

    brief = write(topic, notes)
    for i in range(MAX_REVISIONS):
        verdict = critique(brief, notes)
        print(f"[critic] round {i + 1}: {verdict[:60]}")
        if verdict.startswith("APPROVED"):
            break
        brief = write(topic, notes, feedback=verdict, draft=brief)
    return brief


if __name__ == "__main__":
    print(run(" ".join(sys.argv[1:]) or "How do vector databases work?"))
