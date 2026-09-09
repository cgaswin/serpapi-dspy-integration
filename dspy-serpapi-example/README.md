# DSPy + SerpApi example

This small project demonstrates the native `dspy.Tool.from_serpapi` integration.
It performs real web, news, Maps, shopping, and image searches through SerpApi
and prints compact Markdown results with source links. An optional ReAct mode
lets DSPy choose among the same tools. The hardcoded scenario is a coffee-lover
gift plan for Bangalore, India.

## Setup

1. Install Python 3.10+ and [uv](https://docs.astral.sh/uv/).
2. Copy `.env.example` to `.env` and set `SERPAPI_API_KEY`.
3. Install dependencies:

   ```bash
   uv sync
   ```

4. In a second terminal, start MLflow:

   ```bash
   make mlflow
   ```

To remove all traces from the demo experiment and start a fresh recording:

```bash
make clear-traces
```

The sample points at the two sibling checkouts through `tool.uv.sources`, so it
runs against the implementation in this workspace. For an independent copy,
remove that section after the DSPy and `serpapi-search-tools` releases are
available; the declared version ranges are the published requirements.

## Direct search

```bash
make run
```

This path uses DSPy for tool construction and validation; SerpApi performs five
searches (web, news, Maps, shopping, and images) and formats each result. The
API key is read from the environment and is never passed as a model-visible
argument. The queries are intentionally hardcoded in `main.py` so every
recording follows the same flow.

## ReAct agent

Set `DSPY_MODEL` to the provider-prefixed model identifier used by your DSPy
installation, then run:

```bash
make agent
```

Or pass `--model` explicitly. The agent has six bounded iterations and can select
any of the five native SerpApi tools. JSON output mode is enabled for the agent
so model responses are not sensitive to joined ChatAdapter field markers.

## What this demonstrates

- The SerpApi shared schema is exposed as a native `dspy.Tool`.
- One DSPy program can select web, news, Maps, shopping, and image capabilities.
- DSPy validates tool arguments before dispatch.
- SerpApi applies engine-specific request and response rules.
- `acall()` uses DSPy's bounded worker pool for synchronous HTTP clients.
- Credentials stay in `.env` and outside the model-facing tool schema.
- MLflow records the DSPy/ReAct trace in the `dspy-serpapi-demo` experiment.

Do not commit `.env`. Retries, rate limits, and caching remain application-level
policies.
