"""
agent/middleware.py

Hooks that wrap every tool call and every model call the agent makes.
They run automatically around the agent loop - the model never sees
them and cannot skip them.
"""

import time

from langchain.agents.middleware import wrap_model_call, wrap_tool_call

from src.reliability import task_logger


@wrap_tool_call
def logging_middleware(request, handler):
    """Times each tool call and records it (and its outcome) to the task log."""
    call = request.tool_call
    started = time.perf_counter()
    try:
        response = handler(request)
    except Exception as e:
        task_logger.log_tool_call(
            call["name"], call["args"], f"{type(e).__name__}: {e}",
            time.perf_counter() - started, status="error",
        )
        raise
    task_logger.log_tool_call(
        call["name"], call["args"], getattr(response, "content", str(response)),
        time.perf_counter() - started,
    )
    return response


@wrap_model_call
def model_call_counter(request, handler):
    """Counts every time the agent's reasoning model is called (one per agent step)."""
    task_logger.count_model_call()
    return handler(request)