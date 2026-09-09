"""Run several SerpApi searches or a DSPy ReAct research agent."""

from __future__ import annotations

import argparse
import os
from contextlib import nullcontext

import dspy
from dotenv import load_dotenv
from serpapi_search_tools import images_search, maps_search, news_search, shopping_search

QUESTION = (
    "Help me plan a coffee-lover gift in Bangalore, India: use recent coffee news, "
    "nearby cafe recommendations, product prices, and image results, then cite sources. "
    "For Maps, keep Bangalore in the query and omit the optional location field."
)

SEARCH_QUERIES = {
    "web": "best coffee shops in Bangalore India",
    "news": "Bangalore coffee shop news",
    "maps": "coffee shops in Bangalore India",
    "shopping": "coffee grinder gift India",
    "images": "Bangalore coffee shops",
}


class Research(dspy.Signature):
    """Answer the question using search evidence and include source URLs."""

    question: str = dspy.InputField()
    answer: str = dspy.OutputField()


def build_searches() -> dict[str, dspy.Tool]:
    """Build native DSPy tools for web, news, Maps, shopping, and images."""
    common = {"result_limit": 3, "timeout": 20.0}
    return {
        "web": dspy.Tool.from_serpapi(
            allowed_engines=["google_light"], name="web_search", **common
        ),
        "news": dspy.Tool.from_serpapi(news_search, name="news_search", **common),
        "maps": dspy.Tool.from_serpapi(maps_search, name="maps_search", **common),
        "shopping": dspy.Tool.from_serpapi(
            shopping_search,
            allowed_engines=["google_shopping"],
            name="shopping_search",
            **common,
        ),
        "images": dspy.Tool.from_serpapi(images_search, name="images_search", **common),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--agent", action="store_true", help="Run the DSPy ReAct example"
    )
    parser.add_argument(
        "--model", help="DSPy model identifier, including its provider prefix"
    )
    parser.add_argument(
        "--no-mlflow", action="store_true", help="Run without MLflow tracing"
    )
    args = parser.parse_args()

    load_dotenv()
    if not (os.getenv("SERPAPI_API_KEY") or os.getenv("SERPAPI_KEY")):
        raise SystemExit("Set SERPAPI_API_KEY in .env before running this example.")

    mlflow = None
    if not args.no_mlflow:
        import mlflow

        tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "http://127.0.0.1:5000")
        mlflow.set_tracking_uri(tracking_uri)
        mlflow.set_experiment(os.getenv("MLFLOW_EXPERIMENT", "dspy-serpapi-demo"))
        mlflow.dspy.autolog(log_traces=True, silent=True)
    run_context = (
        mlflow.start_run(run_name="dspy-react" if args.agent else "serpapi-search")
        if mlflow is not None
        else nullcontext()
    )
    with run_context:
        if mlflow is not None:
            mlflow.log_params(
                {
                    "engines": "google_light,google_news,google_maps,google_shopping,google_images",
                    "result_limit": 3,
                    "search_count": len(SEARCH_QUERIES),
                    "mode": "agent" if args.agent else "direct",
                }
            )
        searches = build_searches()
        if not args.agent:
            for kind, query in SEARCH_QUERIES.items():
                print(f"\n## {kind.title()} search: {query}\n")
                print(searches[kind](query=query))
            return

        model = args.model or os.getenv("DSPY_MODEL")
        if not model:
            raise SystemExit("Set DSPY_MODEL or pass --model when using --agent.")
        dspy.configure(lm=dspy.LM(model))
        agent = dspy.ReAct(Research, tools=list(searches.values()), max_iters=6)
        # JSON is less sensitive to models joining ChatAdapter field markers together.
        with dspy.context(adapter=dspy.JSONAdapter()):
            print(agent(question=QUESTION).answer)


if __name__ == "__main__":
    main()
