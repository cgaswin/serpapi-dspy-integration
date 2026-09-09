# Run: uv run --isolated --no-project --with dspy --with serpapi-search-tools --with python-dotenv examples/dspy_openai.py  # noqa: E501
from __future__ import annotations

import os

import dspy
from dotenv import load_dotenv

load_dotenv()

MODEL = os.getenv("DSPY_MODEL", "openai/gpt-4o-mini")


class Research(dspy.Signature):
    """Answer using current web evidence and cite source URLs."""

    question: str = dspy.InputField()
    answer: str = dspy.OutputField()


def _require_serpapi_key() -> None:
    if not (os.getenv("SERPAPI_API_KEY") or os.getenv("SERPAPI_KEY")):
        raise RuntimeError("Set SERPAPI_API_KEY or SERPAPI_KEY before running this example.")


def _require_openai_key() -> None:
    if not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError("Set OPENAI_API_KEY before running this example.")


def main() -> None:
    _require_serpapi_key()
    _require_openai_key()
    dspy.configure(lm=dspy.LM(MODEL))
    search = dspy.Tool.from_serpapi(result_limit=5, timeout=20.0)
    agent = dspy.ReAct(Research, tools=[search], max_iters=4)
    result = agent(question="What changed recently in Python packaging?")
    print(result.answer)


if __name__ == "__main__":
    main()
