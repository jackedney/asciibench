import logging

from asciibench.common.config import Settings

logger = logging.getLogger(__name__)

_LOGFIRE_INITIALIZED = False


def init_logfire(settings: Settings) -> bool:
    """Initialize Logfire observability if enabled in settings.

    Args:
        settings: Application settings containing Logfire configuration

    Returns:
        True if Logfire was initialized successfully, False otherwise

    Example:
        >>> settings = Settings()
        >>> init_logfire(settings)
        True

    Example:
        >>> settings = Settings(logfire=LogfireConfig(enabled=False))
        >>> init_logfire(settings)
        False

    Negative case:
        Invalid token -> logs error, returns False, does not crash application
    """
    global _LOGFIRE_INITIALIZED

    if _LOGFIRE_INITIALIZED:
        return True

    if not settings.logfire.is_enabled:
        return False

    try:
        import logfire

        logfire.configure(
            token=settings.logfire.token,
            service_name=settings.logfire.service_name,
            environment=settings.logfire.environment,
        )

        logfire.instrument_openai()

        _LOGFIRE_INITIALIZED = True
        logger.info(
            f"Logfire initialized: service={settings.logfire.service_name}, "
            f"environment={settings.logfire.environment}"
        )
        return True
    except Exception as e:
        logger.error(f"Failed to initialize Logfire: {e}")
        return False
