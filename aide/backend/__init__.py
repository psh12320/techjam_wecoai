from . import backend_anthropic, backend_openai, backend_openrouter, backend_gemini
from .utils import FunctionSpec, OutputType, PromptType, compile_prompt_to_md
import re
import logging
import os
import threading
import json
import time
from pathlib import Path

logger = logging.getLogger("aide")

_usage_lock = threading.Lock()
_usage_totals = {
    "requests": 0,
    "request_seconds": 0.0,
    "input_tokens": 0,
    "output_tokens": 0,
    "by_model": {},
}
_cost_lock = threading.Lock()
_cost_tracking = None


def configure_cost_tracking(
    path: str | Path,
    input_usd_per_million: float,
    output_usd_per_million: float,
    notification_step_usd: float = 10.0,
    web_search_usd_per_call: float = 0.01,
) -> None:
    """Persist cumulative API cost after every completed request."""

    global _cost_tracking
    if min(
        input_usd_per_million,
        output_usd_per_million,
        web_search_usd_per_call,
    ) < 0:
        raise ValueError("Token prices must be non-negative")
    if notification_step_usd <= 0:
        raise ValueError("notification_step_usd must be positive")
    _cost_tracking = {
        "path": Path(path),
        "input_rate": float(input_usd_per_million),
        "output_rate": float(output_usd_per_million),
        "web_search_rate": float(web_search_usd_per_call),
        "notification_step": float(notification_step_usd),
    }


def _record_cost_event(
    model: str,
    input_tokens: int,
    output_tokens: int,
    *,
    web_search_calls: int = 0,
) -> dict:
    if _cost_tracking is None:
        return {}
    request_cost = (
        input_tokens * _cost_tracking["input_rate"]
        + output_tokens * _cost_tracking["output_rate"]
    ) / 1_000_000
    tool_cost = int(web_search_calls) * _cost_tracking["web_search_rate"]
    request_cost += tool_cost
    path = _cost_tracking["path"]
    with _cost_lock:
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.exists():
            state = json.loads(path.read_text(encoding="utf-8"))
        else:
            state = {
                "total_estimated_cost_usd": 0.0,
                "total_input_tokens": 0,
                "total_output_tokens": 0,
                "total_requests": 0,
                "total_web_search_calls": 0,
                "events": [],
            }
        previous = float(state.get("total_estimated_cost_usd", 0.0))
        current = previous + request_cost
        step = _cost_tracking["notification_step"]
        first_threshold = (int(previous // step) + 1) * step
        crossed = []
        threshold = first_threshold
        while threshold <= current + 1e-12:
            crossed.append(threshold)
            threshold += step
        event = {
            "created_at_unix": time.time(),
            "model": model,
            "input_tokens": int(input_tokens),
            "output_tokens": int(output_tokens),
            "web_search_calls": int(web_search_calls),
            "web_search_cost_usd": tool_cost,
            "estimated_cost_usd": request_cost,
            "cumulative_estimated_cost_usd": current,
            "crossed_notification_thresholds_usd": crossed,
        }
        state["total_estimated_cost_usd"] = current
        state["total_input_tokens"] = int(state.get("total_input_tokens", 0)) + int(
            input_tokens
        )
        state["total_output_tokens"] = int(state.get("total_output_tokens", 0)) + int(
            output_tokens
        )
        state["total_requests"] = int(state.get("total_requests", 0)) + 1
        state["total_web_search_calls"] = int(
            state.get("total_web_search_calls", 0)
        ) + int(web_search_calls)
        state["next_notification_usd"] = (int(current // step) + 1) * step
        state.setdefault("events", []).append(event)
        temporary = path.with_suffix(path.suffix + ".tmp")
        temporary.write_text(
            json.dumps(state, indent=2, sort_keys=True), encoding="utf-8"
        )
        temporary.replace(path)
    for value in crossed:
        print(
            "API_COST_THRESHOLD_CROSSED="
            + json.dumps(
                {"threshold_usd": value, "cumulative_estimated_cost_usd": current}
            )
        )
    return event


def get_cost_tracking_totals() -> dict:
    """Return the durable cumulative cost state, if configured."""

    if _cost_tracking is None or not _cost_tracking["path"].exists():
        return {}
    with _cost_lock:
        return json.loads(_cost_tracking["path"].read_text(encoding="utf-8"))


def reset_usage_totals() -> None:
    """Reset process-local LLM accounting before a new experiment run."""

    with _usage_lock:
        _usage_totals.update(
            requests=0,
            request_seconds=0.0,
            input_tokens=0,
            output_tokens=0,
            by_model={},
        )


def get_usage_totals() -> dict:
    """Return a detached snapshot suitable for challenge resource logs."""

    with _usage_lock:
        return {
            "requests": _usage_totals["requests"],
            "request_seconds": _usage_totals["request_seconds"],
            "input_tokens": _usage_totals["input_tokens"],
            "output_tokens": _usage_totals["output_tokens"],
            "by_model": {
                model: dict(values)
                for model, values in _usage_totals["by_model"].items()
            },
        }


def determine_provider(model: str) -> str:
    # Check if model matches OpenAI patterns first
    if re.match(r"^(gpt-.*|o\d+(-.*)?|codex-mini-latest)$", model):
        return "openai"
    elif model.startswith("claude-"):
        return "anthropic"
    elif model.startswith("gemini-"):
        return "gemini"
    # If OPENAI_BASE_URL is set, use openai provider for non-standard models
    elif os.getenv("OPENAI_BASE_URL"):
        return "openai"
    # all other models are handle by openrouter
    else:
        return "openrouter"


provider_to_query_func = {
    "openai": backend_openai.query,
    "anthropic": backend_anthropic.query,
    "openrouter": backend_openrouter.query,
    "gemini": backend_gemini.query,
}


def query(
    system_message: PromptType | None,
    user_message: PromptType | None,
    model: str,
    temperature: float | None = None,
    max_tokens: int | None = None,
    func_spec: FunctionSpec | None = None,
    **model_kwargs,
) -> OutputType:
    """
    General LLM query for various backends with a single system and user message.
    Supports function calling for some backends.

    Args:
        system_message (PromptType | None): Uncompiled system message (will generate a message following the OpenAI/Anthropic format)
        user_message (PromptType | None): Uncompiled user message (will generate a message following the OpenAI/Anthropic format)
        model (str): string identifier for the model to use (e.g. "gpt-4-turbo")
        temperature (float | None, optional): Temperature to sample at. Defaults to the model-specific default.
        max_tokens (int | None, optional): Maximum number of tokens to generate. Defaults to the model-specific max tokens.
        func_spec (FunctionSpec | None, optional): Optional FunctionSpec object defining a function call. If given, the return value will be a dict.

    Returns:
        OutputType: A string completion if func_spec is None, otherwise a dict with the function call details.
    """

    model_kwargs = model_kwargs | {
        "model": model,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }

    provider = determine_provider(model)
    query_func = provider_to_query_func[provider]
    output, req_time, in_tok_count, out_tok_count, info = query_func(
        system_message=compile_prompt_to_md(system_message) if system_message else None,
        user_message=compile_prompt_to_md(user_message) if user_message else None,
        func_spec=func_spec,
        **model_kwargs,
    )

    with _usage_lock:
        _usage_totals["requests"] += 1
        _usage_totals["request_seconds"] += float(req_time)
        _usage_totals["input_tokens"] += int(in_tok_count)
        _usage_totals["output_tokens"] += int(out_tok_count)
        by_model = _usage_totals["by_model"].setdefault(
            model,
            {
                "requests": 0,
                "request_seconds": 0.0,
                "input_tokens": 0,
                "output_tokens": 0,
            },
        )
        by_model["requests"] += 1
        by_model["request_seconds"] += float(req_time)
        by_model["input_tokens"] += int(in_tok_count)
        by_model["output_tokens"] += int(out_tok_count)

    _record_cost_event(
        model,
        int(in_tok_count),
        int(out_tok_count),
        web_search_calls=int(info.get("web_search_calls", 0) or 0),
    )

    return output
