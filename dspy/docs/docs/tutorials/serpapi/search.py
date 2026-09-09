"""Run a DSPy research agent with SerpApi web search."""

import argparse
import os

import dspy


def create_search_tool(*, client=None, api_key=None):
    """Create a web-search tool; an optional client enables offline testing."""
    return dspy.Tool.from_serpapi(
        allowed_engines=["google_light"],
        result_limit=5,
        timeout=20.0,
        client=client,
        api_key=api_key,
    )


class Research(dspy.Signature):
    """Answer using web search evidence. Cite source URLs and acknowledge missing evidence."""

    question: str = dspy.InputField()
    answer: str = dspy.OutputField()


def _require_serpapi_key() -> str:
    api_key = os.getenv("SERPAPI_API_KEY") or os.getenv("SERPAPI_KEY")
    if not api_key:
        raise SystemExit("Set SERPAPI_API_KEY or SERPAPI_KEY before running this tutorial.")
    return api_key


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("question")
    parser.add_argument("--model", required=True, help="DSPy model identifier, including its provider prefix")
    args = parser.parse_args()

    api_key = _require_serpapi_key()
    dspy.configure(lm=dspy.LM(args.model))
    agent = dspy.ReAct(Research, tools=[create_search_tool(api_key=api_key)], max_iters=4)
    print(agent(question=args.question).answer)


if __name__ == "__main__":
    main()
