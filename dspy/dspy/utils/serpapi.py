"""Bridge SerpApi's public search constructors into DSPy tools."""

import functools
import inspect
from typing import Any, Callable

import anyio

from dspy.adapters.types.tool import Tool
from dspy.utils.asyncify import get_limiter

_DEFAULT_TIMEOUT = 20.0


def _build_search(factory: Callable, constructor_kwargs: dict[str, Any]) -> Any:
    search = factory(provider="dspy", **constructor_kwargs)
    if not callable(search):
        raise TypeError("The SerpApi constructor must return a callable for provider='dspy'.")
    return search


def _search_components(search: Any) -> tuple[Callable, dict[str, Any]]:
    if isinstance(search, Tool):
        return search.func, {
            "name": search.name,
            "desc": search.desc,
            "args": search.args,
            "arg_types": search.arg_types,
            "arg_desc": search.arg_desc,
        }
    if callable(search):
        return search, {}
    raise TypeError("The SerpApi constructor must return a callable for provider='dspy'.")


class _SerpApiTool(Tool):
    def __init__(self, search: Any, *, factory: Callable, constructor_kwargs: dict[str, Any]):
        func, metadata = _search_components(search)
        super().__init__(func, **metadata)
        self._serpapi_factory = factory
        self._serpapi_constructor_kwargs = constructor_kwargs

    async def acall(self, **kwargs: Any) -> str:
        # The public sync path preserves validation, callbacks, and error behavior.
        call = functools.partial(self, **kwargs)
        # Keep the limiter slot until the synchronous request finishes; abandoning a
        # cancelled worker would let new requests exceed async_max_workers.
        return await anyio.to_thread.run_sync(call, abandon_on_cancel=False, limiter=get_limiter())

    def __getstate__(self):
        state = super().__getstate__()
        state["__dict__"] = state["__dict__"].copy()
        state["__dict__"]["func"] = None
        return state

    def __setstate__(self, state):
        state["__dict__"] = state["__dict__"].copy()
        data = state["__dict__"]
        factory = data["_serpapi_factory"]
        constructor_kwargs = data["_serpapi_constructor_kwargs"]
        func, _ = _search_components(_build_search(factory, constructor_kwargs))
        data["func"] = func
        super().__setstate__(state)


def create_serpapi_tool(factory: Callable | None = None, **kwargs: Any) -> Tool:
    if "provider" in kwargs:
        raise ValueError("DSPy selects the SerpApi provider automatically; omit 'provider'.")
    if factory is None:
        try:
            from serpapi_search_tools import web_search
        except ModuleNotFoundError as exc:
            if exc.name != "serpapi_search_tools":
                raise
            raise ImportError(
                "SerpApi search tools are required. Install them with `pip install 'dspy[serpapi]'`."
            ) from exc
        factory = web_search

    if not callable(factory) or "provider" not in inspect.signature(factory).parameters:
        raise TypeError("Pass a SerpApi constructor such as web_search or news_search, not an already-created tool.")
    constructor_kwargs = dict(kwargs)
    if constructor_kwargs.get("timeout") is None:
        constructor_kwargs["timeout"] = _DEFAULT_TIMEOUT
    search = _build_search(factory, constructor_kwargs)
    return _SerpApiTool(search, factory=factory, constructor_kwargs=constructor_kwargs)
