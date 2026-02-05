"""Tests for the observability module."""

import sys
from unittest.mock import MagicMock

from asciibench.common import observability
from asciibench.common.config import LogfireConfig, Settings


class TestInitLogfire:
    """Tests for init_logfire function."""

    def setup_method(self):
        """Reset global state before each test."""
        observability._LOGFIRE_INITIALIZED = False
        if "logfire" in sys.modules:
            del sys.modules["logfire"]

    def test_init_logfire_with_valid_config_returns_true(self):
        """init_logfire returns True when Logfire token is present."""
        mock_logfire = MagicMock()
        mock_logfire.configure = MagicMock()
        mock_logfire.instrument_openai = MagicMock()
        sys.modules["logfire"] = mock_logfire

        settings = Settings(
            logfire=LogfireConfig(token="test-token-123", service_name="test-service")
        )

        result = observability.init_logfire(settings)

        assert result is True
        mock_logfire.configure.assert_called_once_with(
            token="test-token-123",
            service_name="test-service",
            environment="development",
        )
        mock_logfire.instrument_openai.assert_called_once()

    def test_init_logfire_with_disabled_config_returns_false(self):
        """init_logfire returns False when Logfire token is None."""
        settings = Settings(logfire=LogfireConfig(token=None))

        result = observability.init_logfire(settings)

        assert result is False

    def test_init_logfire_with_missing_token_returns_false(self):
        """init_logfire returns False when token is None."""
        settings = Settings(logfire=LogfireConfig(token=None))

        result = observability.init_logfire(settings)

        assert result is False

    def test_init_logfire_with_empty_token(self):
        """init_logfire with empty token returns False."""
        settings = Settings(logfire=LogfireConfig(token=""))

        result = observability.init_logfire(settings)

        assert result is False

    def test_init_logfire_is_idempotent(self):
        """Multiple calls to init_logfire don't cause errors."""
        mock_logfire = MagicMock()
        mock_logfire.configure = MagicMock()
        mock_logfire.instrument_openai = MagicMock()
        sys.modules["logfire"] = mock_logfire

        settings = Settings(
            logfire=LogfireConfig(token="test-token-123", service_name="test-service")
        )

        result1 = observability.init_logfire(settings)
        result2 = observability.init_logfire(settings)
        result3 = observability.init_logfire(settings)

        assert result1 is True
        assert result2 is True
        assert result3 is True

        mock_logfire.configure.assert_called_once()
        mock_logfire.instrument_openai.assert_called_once()

    def test_init_logfire_with_custom_environment(self):
        """init_logfire uses custom environment from config."""
        mock_logfire = MagicMock()
        mock_logfire.configure = MagicMock()
        mock_logfire.instrument_openai = MagicMock()
        sys.modules["logfire"] = mock_logfire

        settings = Settings(
            logfire=LogfireConfig(
                token="test-token-123",
                service_name="test-service",
                environment="production",
            )
        )

        result = observability.init_logfire(settings)

        assert result is True
        mock_logfire.configure.assert_called_once_with(
            token="test-token-123",
            service_name="test-service",
            environment="production",
        )

    def test_init_logfire_handles_exception_gracefully(self):
        """init_logfire logs error and returns False on exception."""
        mock_logfire = MagicMock()
        mock_logfire.configure.side_effect = Exception("Connection error")
        sys.modules["logfire"] = mock_logfire

        settings = Settings(
            logfire=LogfireConfig(token="test-token-123", service_name="test-service")
        )

        result = observability.init_logfire(settings)

        assert result is False
        mock_logfire.configure.assert_called_once()
        mock_logfire.instrument_openai.assert_not_called()


class TestIsLogfireEnabled:
    """Tests for is_logfire_enabled function."""

    def setup_method(self):
        """Reset global state before each test."""
        observability._LOGFIRE_INITIALIZED = False
        if "logfire" in sys.modules:
            del sys.modules["logfire"]

    def test_is_logfire_enabled_returns_false_initially(self):
        """is_logfire_enabled returns False before initialization."""
        assert observability.is_logfire_enabled() is False

    def test_is_logfire_enabled_returns_true_after_init(self):
        """is_logfire_enabled returns True after successful initialization."""
        mock_logfire = MagicMock()
        mock_logfire.configure = MagicMock()
        mock_logfire.instrument_openai = MagicMock()
        sys.modules["logfire"] = mock_logfire

        settings = Settings(
            logfire=LogfireConfig(token="test-token-123", service_name="test-service")
        )

        observability.init_logfire(settings)

        assert observability.is_logfire_enabled() is True

    def test_is_logfire_enabled_returns_false_after_failed_init(self):
        """is_logfire_enabled returns False after failed initialization."""
        mock_logfire = MagicMock()
        mock_logfire.configure.side_effect = Exception("Connection error")
        sys.modules["logfire"] = mock_logfire

        settings = Settings(
            logfire=LogfireConfig(token="test-token-123", service_name="test-service")
        )

        observability.init_logfire(settings)

        assert observability.is_logfire_enabled() is False

    def test_is_logfire_enabled_returns_false_when_disabled(self):
        """is_logfire_enabled returns False when Logfire token is None."""
        settings = Settings(logfire=LogfireConfig(token=None))

        observability.init_logfire(settings)

        assert observability.is_logfire_enabled() is False
