import runpy
from pathlib import Path

import pytest

import dspy
from dspy.utils.dummies import DummyLM

pytest.importorskip("serpapi_search_tools")
pytestmark = pytest.mark.extra


@pytest.fixture
def example():
    path = Path(__file__).resolve().parents[2] / "docs/docs/tutorials/serpapi/search.py"
    return runpy.run_path(str(path))


@pytest.fixture
def client():
    class RecordingClient:
        def __init__(self):
            self.calls = []

        def search(self, params):
            self.calls.append(dict(params))
            return "# Organic Results\n\n[DSPy tools](https://example.com/tools)\n"

    return RecordingClient()


def test_search_schema_and_request(example, client):
    tool = example["create_search_tool"](client=client)
    schema = tool.format_as_litellm_function_call()["function"]["parameters"]
    assert set(schema["properties"]) == {"query", "engine"}
    assert schema["required"] == ["query"]
    assert schema["properties"]["engine"]["enum"] == ["google_light"]
    assert client.calls == []

    result = tool(query="DSPy tools")
    assert "https://example.com/tools" in result
    assert client.calls == [{"q": "DSPy tools", "engine": "google_light", "output": "md"}]


def test_react_uses_search_evidence(example, client):
    tool = example["create_search_tool"](client=client)
    lm = DummyLM(
        [
            {
                "next_thought": "Search the documentation.",
                "next_tool_name": "web_search",
                "next_tool_args": {"query": "DSPy tools"},
            },
            {"next_thought": "I have evidence.", "next_tool_name": "finish", "next_tool_args": {}},
            {"reasoning": "The search returned documentation.", "answer": "See https://example.com/tools"},
        ]
    )
    with dspy.context(lm=lm):
        result = dspy.ReAct(example["Research"], tools=[tool], max_iters=4)(question="How do DSPy tools work?")

    assert len(client.calls) == 1
    assert "https://example.com/tools" in result.trajectory["observation_0"]
    assert "https://example.com/tools" in result.answer


@pytest.mark.asyncio
async def test_async_react_uses_native_bridge(example, client):
    lm = DummyLM(
        [
            {"next_thought": "Search.", "next_tool_name": "web_search", "next_tool_args": {"query": "DSPy"}},
            {"next_thought": "Finish.", "next_tool_name": "finish", "next_tool_args": {}},
            {"reasoning": "Found evidence.", "answer": "See https://example.com/tools"},
        ]
    )
    tool = dspy.Tool.from_serpapi(client=client)
    with dspy.context(lm=lm):
        result = await dspy.ReAct(example["Research"], tools=[tool], max_iters=4).acall(question="What is DSPy?")
    assert len(client.calls) == 1
    assert "https://example.com/tools" in result.trajectory["observation_0"]
