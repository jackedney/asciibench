"""HTMX error handling utilities for judge_ui endpoints.

This module provides decorators for handling exceptions in HTMX endpoints
and rendering appropriate error templates.
"""

import copy
import inspect
import logging
from collections.abc import Callable
from functools import wraps

from fastapi import Request
from fastapi.responses import HTMLResponse


def _get_request_from_args(args: tuple, kwargs: dict) -> Request:
    """Extract the Request object from function arguments.

    FastAPI passes the Request as the first positional argument.
    """
    if "request" in kwargs:
        return kwargs["request"]
    if args and isinstance(args[0], Request):
        return args[0]
    # Fallback - this should not happen in normal use
    raise ValueError("Could not extract Request from function arguments")


def htmx_error_handler(
    template_name: str,
    error_context_key: str = "error",
    error_message_prefix: str = "",
    custom_error_message: str | None = None,
) -> Callable:
    """Decorator to handle exceptions in HTMX endpoints.

    Wraps an endpoint function in a try/except block. On exception,
    renders the specified template with the error message.

    Args:
        template_name: The template path to render on error
        error_context_key: The context key to use for the error message (default: "error")
        error_message_prefix: Optional prefix to add to the error message
        custom_error_message: Optional custom error message instead of exception

    Returns:
        A decorator function

    Example:
        @app.get("/htmx/matchup", response_class=HTMLResponse)
        @htmx_error_handler("partials/matchup.html")
        async def htmx_get_matchup(request: Request) -> HTMLResponse:
            # Endpoint logic here
            return templates.TemplateResponse("partials/matchup.html", {...})
    """

    def decorator(func: Callable) -> Callable:
        @wraps(func)
        async def wrapper(*args, **kwargs) -> HTMLResponse:
            try:
                result = func(*args, **kwargs)
                if inspect.isawaitable(result):
                    result = await result
                return result
            except Exception as e:
                func_name = getattr(func, "__name__", str(func))
                logging.exception(f"Error in {func_name}: {e}")
                from asciibench.judge_ui.main import templates

                request = _get_request_from_args(args, kwargs)
                if custom_error_message:
                    error_message = custom_error_message
                elif error_message_prefix:
                    error_message = f"{error_message_prefix}{e}"
                else:
                    error_message = str(e)
                return templates.TemplateResponse(
                    request,
                    template_name,
                    {error_context_key: error_message},
                )

        return wrapper

    return decorator


def htmx_error_handler_with_context(
    template_name: str,
    default_context: dict,
    error_context_key: str = "error",
    custom_error_message: str | None = None,
) -> Callable:
    """Decorator to handle exceptions in HTMX endpoints with default context.

    Similar to htmx_error_handler, but allows providing a default context
    to merge with the error context on error.

    Args:
        template_name: The template path to render on error
        default_context: Default context to use when rendering error template
        error_context_key: The context key to use for the error message
        custom_error_message: Optional custom error message instead of exception

    Returns:
        A decorator function

    Example:
        @app.get("/htmx/analytics", response_class=HTMLResponse)
        @htmx_error_handler_with_context(
            "partials/analytics.html",
            {"leaderboard": [], "stability": {}, ...},
            custom_error_message="An error occurred. Please try again."
        )
        async def htmx_get_analytics(request: Request) -> HTMLResponse:
            # Endpoint logic here
            return templates.TemplateResponse("partials/analytics.html", {...})
    """

    def decorator(func: Callable) -> Callable:
        @wraps(func)
        async def wrapper(*args, **kwargs) -> HTMLResponse:
            try:
                result = func(*args, **kwargs)
                if inspect.isawaitable(result):
                    result = await result
                return result
            except Exception as e:
                func_name = getattr(func, "__name__", str(func))
                logging.exception(f"Error in {func_name}: {e}")
                from asciibench.judge_ui.main import templates

                request = _get_request_from_args(args, kwargs)
                error_context = copy.deepcopy(default_context)
                if custom_error_message:
                    error_context[error_context_key] = custom_error_message
                else:
                    error_context[error_context_key] = str(e)
                return templates.TemplateResponse(
                    request,
                    template_name,
                    error_context,
                )

        return wrapper

    return decorator
