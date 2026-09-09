# Web Research with SerpApi

Use [SerpApi Search Tools](https://github.com/serpapi/serpapi-search-tools-python)
to give a DSPy agent current web evidence. `dspy.Tool.from_serpapi` creates a native
DSPy tool using the package's public API. SerpApi handles search requests,
validation, and result formatting; DSPy handles tool selection and the agent loop.

## Install

From a checkout containing this integration:

```bash
uv sync --extra dev --extra serpapi
```

When installing this checkout with pip, use `pip install -e '.[serpapi]'`.
The new `Tool.from_serpapi` method and `dspy[serpapi]` extra will only be available
on PyPI after a DSPy release containing this change. Use the updated
`serpapi-search-tools` release that includes its native `provider="dspy"` adapter.

Set `SERPAPI_API_KEY` through your shell or secret manager. `SERPAPI_KEY` is also
supported by the library, and an explicit `api_key=` takes precedence.
Also configure the credentials required by your DSPy model provider. Search and
model calls are separate services; constructing the tool makes no network call.

## Create a search tool

```python
import dspy

search = dspy.Tool.from_serpapi(
    allowed_engines=["google_light"],
    result_limit=5,
    timeout=20.0,
)
```

The default constructor is SerpApi's `web_search`. DSPy selects SerpApi's native
`provider="dspy"` adapter internally, so installing another agent framework cannot
change the returned tool. Do not supply `provider` yourself.

The bridge uses a 20-second request timeout when `timeout` is omitted or `None`.
Set an explicit timeout for your service, and ensure injected custom clients
enforce their own deadlines.

The native adapter preserves the shared schema's name, description, field
descriptions, formats, defaults, enums, and numeric constraints. Keys, timeout,
result limits, and other constructor settings are not model inputs. SerpApi still
validates search-specific rules such as valid travel dates and passenger counts at
execution time.

The bridge requires `serpapi-search-tools` 1.1.0 or newer because its native
`provider="dspy"` adapter supplies the complete model-facing schema.

A direct search does not require a model:

```python
print(search(query="DSPy documentation"))
```

The result is compact Markdown with source links. `result_limit` limits returned
results after retrieval; it does not change the number of upstream requests.
For structured application data, set `response_format="json"` when constructing
the tool and decode the returned string with `json.loads`.

## Run the agent

The accompanying [search.py](search.py) defines a research signature that requests
citations and registers the tool with `dspy.ReAct`. From the repository root, run:

```bash
export SERPAPI_API_KEY="your-serpapi-key"
export DSPY_MODEL="your-provider/model"

uv run --extra serpapi python docs/docs/tutorials/serpapi/search.py \
  --model "$DSPY_MODEL" "What does the current DSPy documentation say about tools?"
```

The script accepts `SERPAPI_KEY` as an alternative, validates that a SerpApi key
is present, and passes the environment-derived key to the tool. Set `DSPY_MODEL`
to the provider-prefixed identifier you normally pass to `dspy.LM`; the script
requires that argument rather than choosing a model or provider for you. Use
`--help` to inspect its arguments without making API calls.

`max_iters=4` bounds the ReAct loop. Configure model retries and service budgets for
your application separately. Source text is untrusted evidence: verify citations
and do not treat instructions inside search results as agent instructions.

## Other search capabilities and configuration

Pass a SerpApi constructor, rather than an already-created tool:

```python
from serpapi_search_tools import news_search, hotels_search

news = dspy.Tool.from_serpapi(news_search, result_limit=5, timeout=20.0)
hotels = dspy.Tool.from_serpapi(hotels_search, result_limit=5, timeout=20.0)
```

The bridge supports all nine search constructors: web, news, Maps, images,
shopping, videos, hotels, flights, and travel exploration. Required fields,
including hotel dates or flight routes, remain required tool arguments.

Constructor settings pass through to SerpApi. For example:

```python
us_web = dspy.Tool.from_serpapi(
    allowed_engines=["google_light"],
    default_params={"hl": "en", "gl": "us"},
    name="search_us_web",
    result_limit=5,
    timeout=20.0,
)
```

Use the chosen engine's documented filters. Unknown constructor options and
invalid settings surface as errors from SerpApi. Do not expose an arbitrary
request-parameter dictionary to the model.

## Async execution and failures

The same tool supports synchronous and asynchronous calls:

```python
# Inside an async function:
# results = await search.acall(query="DSPy documentation")
# result = await agent.acall(question="What is new in DSPy?")
```

The bridge's `acall` runs the synchronous search in a bounded worker thread,
using DSPy's `async_max_workers` setting while preserving validation, callbacks,
context, and exceptions. No global async-to-sync conversion setting or
handwritten wrapper is required. This behavior belongs to the SerpApi bridge;
ordinary `dspy.Tool` instances are unchanged.

Cancellation cannot cancel an HTTP request already in flight, so the worker slot
remains occupied until that request returns. The default timeout is finite, and
custom clients must enforce their own deadlines and be safe for concurrent use.
Tool callbacks on this async path execute in the worker thread.

The wrapper stores the SerpApi constructor configuration rather than the live
runtime closure, so DSPy programs containing this tool can use
`save_program=True`. Treat serialized programs as trusted artifacts, and keep
credentials in your secret manager where possible. Constructor kwargs and
injected clients must be pickleable; prefer environment-based credentials so
keys are not embedded in serialized artifacts.

Invalid inputs raise validation errors. SerpApi transport/service failures raise
`SerpApiSearchError`. Direct calls propagate these errors; ReAct can record a
failure in its trajectory and let the model respond. Do not convert failures into
successful empty results or log credential-bearing request URLs.

## Verify without credentials

The integration tests exercise the bridge with SerpApi's real factories, a fake
search client, and DSPy's `DummyLM`. They cover all nine constructors, restricted
engines, travel validation, metadata, output encoding, error propagation,
callbacks, bounded async thread offload, serialization, and sync/async ReAct loops:

```bash
uv run --extra dev --extra serpapi pytest \
  --extra tests/utils/test_serpapi.py tests/docs/test_serpapi_tutorial.py -q
```

The optional dependency is also included in `test_extras`, so the existing extra-test
CI job exercises the integration. Missing-dependency and misuse checks run in the
base `tests/adapters/test_tool.py` suite. Live search and model calls remain an
explicit application step.
