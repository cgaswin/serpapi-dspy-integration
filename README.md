# DSPy + SerpApi integration

This repository is a self-contained snapshot of the native DSPy integration for
[SerpApi Search Tools](https://github.com/serpapi/serpapi-search-tools-python).
SerpApi owns search semantics, validation, and result formatting; DSPy owns tool
schemas, orchestration, async execution, and ReAct.

## What is included

- [`dspy/`](dspy/) — DSPy 3.3.1 with `Tool.from_serpapi`, the async-capable
  bridge, optional `serpapi` extra, tutorial, and tests.
- [`serpapi-search-tools-python/`](serpapi-search-tools-python/) — SerpApi
  Search Tools with the native `provider="dspy"` adapter and its tests/docs.
- [`dspy-serpapi-example/`](dspy-serpapi-example/) — runnable direct-search and
  bounded ReAct demo using web, news, Maps, shopping, and image tools.

The bridge supports all nine SerpApi constructors (web, news, Maps, images,
shopping, videos, hotels, flights, and travel exploration). Constructor settings
stay application-controlled; only the typed search inputs are exposed to a
model. `acall()` runs synchronous clients in DSPy's bounded worker pool, and
serialized tools rebuild from their constructor configuration.

## Quick start

From the repository root:

```bash
cd dspy-serpapi-example
cp .env.example .env
# Set SERPAPI_API_KEY in .env; never commit that file.
uv sync
make run
```

For the ReAct demo, also set a provider-prefixed model identifier:

```bash
export DSPY_MODEL="provider/model"
make agent
```

MLflow tracing is enabled by default. Start `make mlflow` in another terminal,
or pass `--no-mlflow` to `main.py`.

The public API is:

```python
import dspy
from serpapi_search_tools import news_search

web = dspy.Tool.from_serpapi(result_limit=5, timeout=20.0)
news = dspy.Tool.from_serpapi(news_search, result_limit=5, timeout=20.0)
print(web(query="DSPy documentation"))
```

## Offline checks

The DSPy tests use fake SerpApi clients and `DummyLM`; they do not need API keys
or live model calls. From `dspy/`, install the dev and SerpApi extras and run:

```bash
uv sync --extra dev --extra serpapi
uv run pytest tests/utils/test_serpapi.py tests/docs/test_serpapi_tutorial.py -q
```

The SerpApi adapter tests can be run from `serpapi-search-tools-python/` with:

```bash
uv sync --extra dev
uv run pytest tests/test_adapters.py -q
```

## Security and scope

Keep `SERPAPI_API_KEY`, model credentials, `.env`, MLflow state, and generated
virtual environments out of commits. Search results are untrusted evidence and
should not be treated as agent instructions. Retries, rate limits, and caching
remain application-level policies.

The two narrative files used during development (`presentation.md` and
`integration.md`) are intentionally not part of this repository.
