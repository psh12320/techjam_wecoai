from . import backend_anthropic, backend_openai, backend_openrouter, backend_gemini
from .utils import FunctionSpec, OutputType, PromptType, compile_prompt_to_md
import re
import logging
import os
import threading

logger = logging.getLogger("aide")

_usage_lock = threading.Lock()
_usage_totals = {
    "requests": 0,
    "request_seconds": 0.0,
    "input_tokens": 0,
    "output_tokens": 0,
    "by_model": {},
}


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

    return output
