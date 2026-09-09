import asyncio

import pytest

import dspy

serpapi_search_tools = pytest.importorskip("serpapi_search_tools")
pytestmark = pytest.mark.extra


@pytest.fixture
def client():
    class RecordingClient:
        def __init__(self):
            self.calls = []

        def search(self, params):
            self.calls.append(dict(params))
            return "# Organic Results\n\n[DSPy tools](https://example.com/tools)\n"

    return RecordingClient()


@pytest.mark.parametrize(
    "arguments", [{"query": " "}, {"query": "DSPy", "engine": "bing"}, {"query": "DSPy", "api_key": "not-a-credential"}]
)
def test_invalid_inputs_do_not_search(client, arguments):
    tool = dspy.Tool.from_serpapi(
        allowed_engines=["google_light"], result_limit=5, timeout=20.0, client=client
    )
    with pytest.raises(ValueError):
        tool(**arguments)
    assert client.calls == []


def test_service_failure_propagates_without_client_details():
    class FailingClient:
        def search(self, params):
            raise RuntimeError("private client diagnostic")

    tool = dspy.Tool.from_serpapi(
        allowed_engines=["google_light"], result_limit=5, timeout=20.0, client=FailingClient()
    )
    with pytest.raises(serpapi_search_tools.SerpApiSearchError) as error:
        tool(query="DSPy tools")
    assert "private client diagnostic" not in str(error.value)


def test_json_search_returns_decodable_results():
    import json

    class JsonClient:
        def search(self, params):
            return {
                "organic_results": [
                    {"title": "First", "link": "https://example.com/first"},
                    {"title": "Second", "link": "https://example.com/second"},
                ]
            }

    tool = dspy.Tool.from_serpapi(client=JsonClient(), response_format="json", result_limit=1)
    result = json.loads(tool(query="DSPy tools"))
    assert result["organic_results"] == [{"title": "First", "link": "https://example.com/first"}]


@pytest.mark.parametrize(
    "factory_name, arguments, expected_engine",
    [
        ("web_search", {"query": "DSPy"}, "google_light"),
        ("news_search", {"query": "DSPy"}, "google_news"),
        ("maps_search", {"query": "coffee"}, "google_maps"),
        ("images_search", {"query": "coffee"}, "google_images"),
        ("shopping_search", {"query": "coffee"}, "google_shopping"),
        ("videos_search", {"query": "DSPy"}, "youtube"),
        (
            "hotels_search",
            {"query": "Kyoto", "check_in_date": "2030-08-01", "check_out_date": "2030-08-04"},
            "google_hotels",
        ),
        (
            "flights_search",
            {"departure_id": "JFK", "arrival_id": "SFO", "outbound_date": "2030-08-01"},
            "google_flights",
        ),
        ("travel_explore_search", {"departure_id": "JFK"}, "google_travel_explore"),
    ],
)
def test_every_search_constructor(factory_name, arguments, expected_engine, client):
    tool = dspy.Tool.from_serpapi(getattr(serpapi_search_tools, factory_name), client=client)
    assert isinstance(tool, dspy.Tool)
    assert tool.name == factory_name
    assert set(arguments).issubset(tool.args)
    tool(**arguments)
    assert len(client.calls) == 1
    assert client.calls[0]["engine"] == expected_engine


def test_travel_defaults_enums_and_runtime_validation(client):
    tool = dspy.Tool.from_serpapi(serpapi_search_tools.hotels_search, client=client)
    schema = tool.format_as_litellm_function_call()["function"]["parameters"]
    assert schema["required"] == ["query", "check_in_date", "check_out_date"]
    assert tool.args["adults"]["default"] == 2
    assert tool.args["children_ages"]["default"] is None
    with pytest.raises(ValueError):
        tool(query="Kyoto", check_in_date="2030-08-04", check_out_date="2030-08-01")
    with pytest.raises(ValueError):
        tool(query="Kyoto", check_in_date="2030-08-01", check_out_date="2030-08-04", adults=0)
    assert client.calls == []

    flight = dspy.Tool.from_serpapi(serpapi_search_tools.flights_search, client=client)
    assert "economy" in flight.args["travel_class"]["enum"]
    assert flight.args["travel_class"]["default"] == "economy"


@pytest.mark.parametrize(
    "factory, arguments",
    [
        (
            serpapi_search_tools.hotels_search,
            {
                "query": "Kyoto",
                "check_in_date": "2030-08-01",
                "check_out_date": "2030-08-04",
                "children_ages": None,
            },
        ),
        (
            serpapi_search_tools.flights_search,
            {
                "departure_id": "JFK",
                "arrival_id": "SFO",
                "outbound_date": "2030-08-01",
                "return_date": None,
            },
        ),
    ],
)
def test_optional_none_values_are_accepted(client, factory, arguments):
    tool = dspy.Tool.from_serpapi(factory, client=client)

    tool(**arguments)

    assert len(client.calls) == 1


def test_native_travel_schema_preserves_constraints_and_descriptions(client):
    tool = dspy.Tool.from_serpapi(serpapi_search_tools.hotels_search, client=client)

    assert tool.args["adults"]["minimum"] == 1
    assert tool.args["check_in_date"]["description"]


def test_constructor_configuration_is_kept_out_of_model_schema(client):
    tool = dspy.Tool.from_serpapi(
        result_limit=2,
        name="current_web",
        default_params={"hl": "en"},
        allowed_engines=["bing"],
        client=client,
    )
    assert tool.name == "current_web"
    assert set(tool.args) == {"query", "engine"}
    tool(query="DSPy")
    assert client.calls[0]["engine"] == "bing"
    assert client.calls[0]["hl"] == "en"


def test_tool_can_be_pickled_and_rebuilt():
    import cloudpickle

    class PickleClient:
        def search(self, params):
            return "# Organic Results\n\n[Result](https://example.com/result)\n"

    tool = dspy.Tool.from_serpapi(client=PickleClient())
    restored = cloudpickle.loads(cloudpickle.dumps(tool))

    assert "https://example.com/result" in tool(query="DSPy")
    assert "https://example.com/result" in restored(query="DSPy")


def test_tool_pickle_preserves_customized_metadata():
    import cloudpickle

    class PickleClient:
        def search(self, params):
            return "# Organic Results\n"

    tool = dspy.Tool.from_serpapi(client=PickleClient())
    tool.name = "custom_name"
    tool.desc = "custom description"
    restored = cloudpickle.loads(cloudpickle.dumps(tool))

    assert restored.name == "custom_name"
    assert restored.desc == "custom description"

    react = dspy.ReAct("question -> answer", tools=[tool])
    restored_react = cloudpickle.loads(cloudpickle.dumps(react))
    assert restored_react.tools["custom_name"].name == "custom_name"


def test_default_timeout_is_forwarded_to_constructor():
    captured = {}

    def factory(*, provider, timeout):
        captured["provider"] = provider
        captured["timeout"] = timeout

        def search(query: str) -> str:
            return query

        return search

    dspy.Tool.from_serpapi(factory)

    assert captured == {"provider": "dspy", "timeout": 20.0}

    dspy.Tool.from_serpapi(factory, timeout=None)
    assert captured["timeout"] == 20.0


def test_native_serpapi_provider_is_required():
    providers = []

    def factory(*, provider, timeout):
        providers.append(provider)
        raise ValueError("Unknown provider 'dspy'.")

    with pytest.raises(ValueError, match="Unknown provider 'dspy'"):
        dspy.Tool.from_serpapi(factory)

    assert providers == ["dspy"]


@pytest.mark.asyncio
async def test_async_search_offloads_and_preserves_callbacks():
    import threading

    from dspy.utils.callback import BaseCallback

    loop_thread = threading.get_ident()
    search_threads = []
    events = []

    class Client:
        def search(self, params):
            search_threads.append(threading.get_ident())
            assert dspy.settings.max_errors == 17
            return "# Organic Results\n\n[DSPy](https://example.com/tools)\n"

    class Callback(BaseCallback):
        def on_tool_start(self, call_id, instance, inputs):
            events.append(("start", call_id, inputs))

        def on_tool_end(self, call_id, outputs, exception=None):
            events.append(("end", call_id, exception))

    tool = dspy.Tool.from_serpapi(client=Client())
    with dspy.context(callbacks=[Callback()], max_errors=17):
        result = await tool.acall(query="DSPy")

    assert "https://example.com/tools" in result
    assert len(search_threads) == 1
    assert search_threads[0] != loop_thread
    assert [event[0] for event in events] == ["start", "end"]
    assert events[0][1] == events[1][1]
    assert events[1][2] is None


@pytest.mark.asyncio
async def test_async_search_honors_dspy_concurrency_limit():
    import threading

    active = 0
    peak = 0
    lock = threading.Lock()

    class Client:
        def search(self, params):
            nonlocal active, peak
            with lock:
                active += 1
                peak = max(peak, active)
            import time

            time.sleep(0.02)
            with lock:
                active -= 1
            return "# Organic Results\n"

    tool = dspy.Tool.from_serpapi(client=Client())
    with dspy.context(async_max_workers=1):
        await asyncio.gather(*(tool.acall(query=str(index)) for index in range(8)))

    assert peak == 1


@pytest.mark.asyncio
async def test_cancelled_async_search_keeps_its_limiter_slot_until_completion():
    import threading
    import time

    active = 0
    peak = 0
    lock = threading.Lock()
    started = threading.Event()

    class Client:
        def search(self, params):
            nonlocal active, peak
            with lock:
                active += 1
                peak = max(peak, active)
                started.set()
            time.sleep(0.08)
            with lock:
                active -= 1
            return "# Organic Results\n"

    tool = dspy.Tool.from_serpapi(client=Client())
    with dspy.context(async_max_workers=1):
        tasks = [asyncio.create_task(tool.acall(query=str(index))) for index in range(4)]
        await asyncio.to_thread(started.wait, 1)
        for task in tasks:
            task.cancel()
        results = await asyncio.gather(*tasks, return_exceptions=True)

    assert all(isinstance(result, asyncio.CancelledError) for result in results)
    assert peak == 1


@pytest.mark.asyncio
async def test_async_errors_and_validation(client):
    tool = dspy.Tool.from_serpapi(client=client)
    with pytest.raises(ValueError):
        await tool.acall(query=" ")
    assert client.calls == []

    class FailingClient:
        def search(self, params):
            raise RuntimeError("private diagnostic")

    failing = dspy.Tool.from_serpapi(client=FailingClient())
    with pytest.raises(serpapi_search_tools.SerpApiSearchError) as error:
        await failing.acall(query="DSPy")
    assert "private diagnostic" not in str(error.value)
