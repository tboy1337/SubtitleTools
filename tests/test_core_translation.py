"""Tests for the translation module."""

import json
import urllib.error
import urllib.request
from unittest.mock import Mock, patch

import pytest
import requests

from subtitletools.core.translation import (
    GoogleTranslator,
    RateLimitError,
    SubtitleTranslator,
    TkGenerator,
    TranslationError,
    Translator,
    get_translator,
    is_space_language,
)


class TestExceptions:
    """Test exception classes."""

    def test_translation_error_creation(self) -> None:
        """Test TranslationError creation."""
        error = TranslationError("Test error")
        assert str(error) == "Test error"
        assert isinstance(error, Exception)

    def test_rate_limit_error_creation(self) -> None:
        """Test RateLimitError creation."""
        error = RateLimitError("Rate limited")
        assert str(error) == "Rate limited"
        assert isinstance(error, TranslationError)
        assert isinstance(error, Exception)

    def test_translation_error_inheritance(self) -> None:
        """Test that RateLimitError inherits from TranslationError."""
        error = RateLimitError("Test")
        assert isinstance(error, TranslationError)


class TestTranslatorABC:
    """Test abstract Translator class."""

    def test_translator_is_abstract(self) -> None:
        """Test that Translator cannot be instantiated directly."""
        with pytest.raises(TypeError, match="Can't instantiate abstract class"):
            Translator()  # type: ignore[abstract]  # pylint: disable=abstract-class-instantiated


class TestTkGenerator:
    """Test TkGenerator class."""

    def test_init(self) -> None:
        """Test TkGenerator initialization."""
        gen = TkGenerator()
        assert gen.js_code is not None
        assert "var tk = function" in gen.js_code

    def test_get_js_code(self) -> None:
        """Test JavaScript code generation."""
        gen = TkGenerator()
        js_code = gen._get_js_code()
        assert "var b = function" in js_code
        assert "var tk = function" in js_code

    @patch('execjs.compile')
    def test_generate_success(self, mock_compile: Mock) -> None:
        """Test successful token generation."""
        mock_ctx = Mock()
        mock_ctx.call.return_value = "123456.789012"
        mock_compile.return_value = mock_ctx

        gen = TkGenerator()
        result = gen.generate("Hello", "123.456")

        assert result == "123456.789012"
        mock_compile.assert_called_once()
        mock_ctx.call.assert_called_once_with("tk", "Hello", "123.456")

    @patch('execjs.compile')
    def test_generate_none_result(self, mock_compile: Mock) -> None:
        """Test token generation with None result."""
        mock_ctx = Mock()
        mock_ctx.call.return_value = None
        mock_compile.return_value = mock_ctx

        gen = TkGenerator()
        result = gen.generate("Hello")

        assert result == "0"

    @patch('execjs.compile')
    def test_generate_exception(self, mock_compile: Mock) -> None:
        """Test token generation with exception."""
        mock_compile.side_effect = Exception("JS error")

        gen = TkGenerator()
        result = gen.generate("Hello")

        assert result == "0"


class TestGoogleTranslator:
    """Test GoogleTranslator class."""

    def test_init_without_api_key(self) -> None:
        """Test GoogleTranslator initialization without API key."""
        translator = GoogleTranslator()
        assert translator.api_key is None
        assert translator.headers["User-Agent"] is not None
        assert translator.tk_gen is not None
        assert translator.max_limited == 3500
        assert translator.max_retries == 5

    def test_init_with_api_key(self) -> None:
        """Test GoogleTranslator initialization with API key."""
        translator = GoogleTranslator(api_key="test_key")
        assert translator.api_key == "test_key"

    def test_rotate_user_agent(self) -> None:
        """Test user agent rotation."""
        translator = GoogleTranslator()

        translator._rotate_user_agent()

        # User agent should be one of the predefined ones
        assert translator.headers["User-Agent"] in translator.user_agents

    def test_calculate_backoff(self) -> None:
        """Test backoff calculation."""
        translator = GoogleTranslator()

        backoff_0 = translator._calculate_backoff(0)
        backoff_1 = translator._calculate_backoff(1)
        backoff_large = translator._calculate_backoff(10)

        # Should increase with retry count
        assert backoff_1 > backoff_0
        # Should be capped at max_backoff
        assert backoff_large <= translator.max_backoff + translator.max_backoff * translator.jitter

    def test_translate_empty_text(self) -> None:
        """Test translation of empty text."""
        translator = GoogleTranslator()
        result = translator.translate("   ", "en", "es")
        assert result == "   "

    @patch('requests.post')
    def test_translate_with_api_success(self, mock_post: Mock) -> None:
        """Test successful API translation."""
        mock_response = Mock()
        mock_response.json.return_value = {
            "data": {
                "translations": [{"translatedText": "Hola mundo"}]
            }
        }
        mock_post.return_value = mock_response

        translator = GoogleTranslator(api_key="test_key")
        result = translator.translate("Hello world", "en", "es")

        assert result == "Hola mundo"
        mock_post.assert_called_once()

    @patch('requests.post')
    def test_translate_with_api_invalid_response(self, mock_post: Mock) -> None:
        """Test API translation with invalid response."""
        mock_response = Mock()
        mock_response.json.return_value = {"error": "Invalid request"}
        mock_post.return_value = mock_response

        translator = GoogleTranslator(api_key="test_key")

        with pytest.raises(TranslationError, match="Invalid API response format"):
            translator.translate("Hello", "en", "es")

    @patch('requests.post')
    def test_translate_with_api_request_exception(self, mock_post: Mock) -> None:
        """Test API translation with request exception."""
        mock_post.side_effect = requests.RequestException("Connection error")

        translator = GoogleTranslator(api_key="test_key")

        with pytest.raises(TranslationError, match="API request failed"):
            translator.translate("Hello", "en", "es")

    def test_translate_with_web_long_text(self) -> None:
        """Test web translation with text too long."""
        translator = GoogleTranslator()
        long_text = "a" * (translator.max_limited + 1)

        with pytest.raises(TranslationError, match="Text too long for web interface"):
            translator.translate(long_text, "en", "es")

    @patch('subtitletools.core.translation.GoogleTranslator._make_request_with_retry')
    def test_translate_with_web_success(self, mock_request: Mock) -> None:
        """Test successful web translation."""
        mock_response = json.dumps([[["Hola mundo", "Hello world", None, None, 10]]])
        mock_request.return_value = mock_response

        translator = GoogleTranslator()
        result = translator.translate("Hello world", "en", "es")

        assert result == "Hola mundo"

    @patch('subtitletools.core.translation.GoogleTranslator._make_request_with_retry')
    def test_translate_with_web_multiline(self, mock_request: Mock) -> None:
        """Test web translation with multiline text."""
        mock_response = json.dumps([[["Hola", "Hello", None, None, 10], ["mundo", "world", None, None, 10]]])
        mock_request.return_value = mock_response

        translator = GoogleTranslator()
        result = translator.translate("Hello\nworld", "en", "es")

        assert result == "Hola\nmundo"

    @patch('subtitletools.core.translation.GoogleTranslator._make_request_with_retry')
    def test_translate_with_web_empty_response(self, mock_request: Mock) -> None:
        """Test web translation with empty response."""
        mock_request.return_value = json.dumps([])

        translator = GoogleTranslator()

        with pytest.raises(TranslationError, match="Empty translation response"):
            translator.translate("Hello", "en", "es")

    @patch('subtitletools.core.translation.GoogleTranslator._make_request_with_retry')
    def test_translate_with_web_json_error(self, mock_request: Mock) -> None:
        """Test web translation with JSON decode error."""
        mock_request.return_value = "invalid json"

        translator = GoogleTranslator()

        with pytest.raises(TranslationError, match="Failed to parse translation response"):
            translator.translate("Hello", "en", "es")

    @patch('urllib.request.urlopen')
    def test_make_request_success(self, mock_urlopen: Mock) -> None:
        """Test successful HTTP request."""
        mock_response = Mock()
        mock_response.read.return_value = b"Success"
        mock_response.__enter__ = Mock(return_value=mock_response)
        mock_response.__exit__ = Mock(return_value=None)
        mock_urlopen.return_value = mock_response

        translator = GoogleTranslator()
        result = translator._make_request_with_retry("http://example.com")

        assert result == "Success"

    @patch('urllib.request.urlopen')
    def test_make_request_rate_limit_retry(self, mock_urlopen: Mock) -> None:
        """Test HTTP request with rate limiting."""
        # First call raises 429, second succeeds
        error = urllib.error.HTTPError("", 429, "Too Many Requests", {}, None)
        mock_response = Mock()
        mock_response.read.return_value = b"Success"
        mock_response.__enter__ = Mock(return_value=mock_response)
        mock_response.__exit__ = Mock(return_value=None)

        mock_urlopen.side_effect = [error, mock_response]

        with patch('time.sleep'):  # Mock sleep to speed up test
            translator = GoogleTranslator()
            translator.max_retries = 2  # Reduce retries for faster test
            result = translator._make_request_with_retry("http://example.com")

        assert result == "Success"

    @patch('urllib.request.urlopen')
    def test_make_request_rate_limit_exceeded(self, mock_urlopen: Mock) -> None:
        """Test HTTP request with rate limit exceeded."""
        error = urllib.error.HTTPError("", 429, "Too Many Requests", {}, None)
        mock_urlopen.side_effect = error

        with patch('time.sleep'):  # Mock sleep to speed up test
            translator = GoogleTranslator()
            translator.max_retries = 2

            with pytest.raises(RateLimitError, match="Rate limit exceeded after"):
                translator._make_request_with_retry("http://example.com")

    @patch('urllib.request.urlopen')
    def test_make_request_server_error(self, mock_urlopen: Mock) -> None:
        """Test HTTP request with server error."""
        error = urllib.error.HTTPError("", 500, "Internal Server Error", {}, None)
        mock_urlopen.side_effect = error

        with patch('time.sleep'):  # Mock sleep to speed up test
            translator = GoogleTranslator()
            translator.max_retries = 1

            with pytest.raises(TranslationError, match="Server error 500"):
                translator._make_request_with_retry("http://example.com")

    @patch('urllib.request.urlopen')
    def test_make_request_http_error(self, mock_urlopen: Mock) -> None:
        """Test HTTP request with other HTTP error."""
        error = urllib.error.HTTPError("", 404, "Not Found", {}, None)
        mock_urlopen.side_effect = error

        translator = GoogleTranslator()

        with pytest.raises(TranslationError, match="HTTP error 404"):
            translator._make_request_with_retry("http://example.com")

    @patch('urllib.request.urlopen')
    def test_make_request_url_error(self, mock_urlopen: Mock) -> None:
        """Test HTTP request with URL error."""
        error = urllib.error.URLError("Connection failed")
        mock_urlopen.side_effect = error

        with patch('time.sleep'):  # Mock sleep to speed up test
            translator = GoogleTranslator()
            translator.max_retries = 1

            with pytest.raises(TranslationError, match="Request failed after"):
                translator._make_request_with_retry("http://example.com")

    def test_translate_lines_empty(self) -> None:
        """Test translation of empty lines."""
        translator = GoogleTranslator()
        result = translator.translate_lines([], "en", "es")
        assert result == ""

    @patch('subtitletools.core.translation.GoogleTranslator.translate')
    def test_translate_lines_success(self, mock_translate: Mock) -> None:
        """Test successful line translation."""
        mock_translate.return_value = "Hola\nmundo"

        translator = GoogleTranslator()
        result = translator.translate_lines(["Hello", "world"], "en", "es")

        assert result == "Hola\nmundo"
        mock_translate.assert_called_once_with("Hello\nworld", "en", "es")

    @patch('subtitletools.core.translation.GoogleTranslator.translate')
    @patch('time.sleep')
    def test_translate_lines_chunked(self, mock_sleep: Mock, mock_translate: Mock) -> None:
        """Test line translation with chunking."""
        # Create text that exceeds max_limited
        translator = GoogleTranslator()
        translator.max_limited = 10  # Small limit for testing

        long_lines = ["This is a very long line that exceeds the limit", "Another long line"]
        mock_translate.side_effect = ["Línea larga 1", "Línea larga 2"]

        result = translator.translate_lines(long_lines, "en", "es")

        # Should be called twice (once per chunk)
        assert mock_translate.call_count == 2
        assert result == "Línea larga 1\nLínea larga 2"


class TestSubtitleTranslator:
    """Test SubtitleTranslator class."""

    def test_init_valid_service(self) -> None:
        """Test initialization with valid service."""
        translator = SubtitleTranslator("google")
        assert translator.service_name == "google"
        assert translator.translator is not None

    def test_init_invalid_service(self) -> None:
        """Test initialization with invalid service."""
        with pytest.raises(TranslationError, match="Unsupported translation service"):
            SubtitleTranslator("invalid_service")

    def test_get_translator_google(self) -> None:
        """Test getting Google translator."""
        translator = SubtitleTranslator("google")
        result = translator._get_translator("google", None)
        assert isinstance(result, GoogleTranslator)

    def test_get_translator_unsupported(self) -> None:
        """Test getting unsupported translator."""
        translator = SubtitleTranslator("google")
        with pytest.raises(TranslationError, match="Unsupported translation service"):
            translator._get_translator("unsupported", None)

    def test_translate_text_empty(self) -> None:
        """Test translation of empty text."""
        translator = SubtitleTranslator("google")
        result = translator.translate_text("   ")
        assert result == "   "

    @patch('subtitletools.core.translation.GoogleTranslator.translate')
    def test_translate_text_success(self, mock_translate: Mock) -> None:
        """Test successful text translation."""
        mock_translate.return_value = "Hola mundo"

        translator = SubtitleTranslator("google")
        result = translator.translate_text("Hello world", "en", "es")

        assert result == "Hola mundo"
        mock_translate.assert_called_once_with("Hello world", "en", "es")

    @patch('subtitletools.core.translation.GoogleTranslator.translate')
    def test_translate_text_with_progress(self, mock_translate: Mock) -> None:
        """Test text translation with progress callback."""
        mock_translate.return_value = "Hola mundo"
        progress_calls = []

        def progress_callback(current: int, total: int, message: str) -> None:
            progress_calls.append((current, total, message))

        translator = SubtitleTranslator("google")
        result = translator.translate_text("Hello world", "en", "es", progress_callback)

        assert result == "Hola mundo"
        assert len(progress_calls) == 2
        assert progress_calls[0] == (0, 1, "Translating...")
        assert progress_calls[1] == (1, 1, "Complete")

    @patch('subtitletools.core.translation.GoogleTranslator.translate')
    def test_translate_text_exception(self, mock_translate: Mock) -> None:
        """Test text translation with exception."""
        mock_translate.side_effect = Exception("Translation failed")

        translator = SubtitleTranslator("google")

        with pytest.raises(TranslationError, match="Translation failed"):
            translator.translate_text("Hello world")

    def test_translate_lines_empty(self) -> None:
        """Test translation of empty lines."""
        translator = SubtitleTranslator("google")
        result = translator.translate_lines([])
        assert not result

    @patch('subtitletools.core.translation.GoogleTranslator.translate_lines')
    def test_translate_lines_success(self, mock_translate_lines: Mock) -> None:
        """Test successful lines translation."""
        mock_translate_lines.return_value = "Hola\nmundo"

        translator = SubtitleTranslator("google")
        result = translator.translate_lines(["Hello", "world"], "en", "es")

        assert result == ["Hola", "mundo"]
        mock_translate_lines.assert_called_once()

    @patch('subtitletools.core.translation.GoogleTranslator.translate_lines')
    def test_translate_lines_with_empty_lines(self, mock_translate_lines: Mock) -> None:
        """Test lines translation with empty lines."""
        mock_translate_lines.return_value = "Hola\nmundo"

        translator = SubtitleTranslator("google")
        result = translator.translate_lines(["Hello", "", "world"], "en", "es")

        # Empty line should be preserved
        assert len(result) == 3
        assert result[1] == ""

    @patch('subtitletools.core.translation.GoogleTranslator.translate_lines')
    def test_translate_lines_exception(self, mock_translate_lines: Mock) -> None:
        """Test lines translation with exception."""
        mock_translate_lines.side_effect = Exception("Translation failed")

        translator = SubtitleTranslator("google")

        with pytest.raises(TranslationError, match="Batch translation failed"):
            translator.translate_lines(["Hello", "world"])

    def test_get_service_info(self) -> None:
        """Test getting service information."""
        translator = SubtitleTranslator("google", "test_key")
        info = translator.get_service_info()

        assert info["service"] == "google"
        assert info["has_api_key"] is True
        assert info["max_text_length"] == 3500

    def test_get_service_info_no_api_key(self) -> None:
        """Test getting service information without API key."""
        translator = SubtitleTranslator("google")
        info = translator.get_service_info()

        assert info["service"] == "google"
        assert info["has_api_key"] is False


class TestUtilityFunctions:
    """Test utility functions."""

    def test_get_translator_google(self) -> None:
        """Test getting Google translator."""
        translator = get_translator("google")
        assert isinstance(translator, GoogleTranslator)

    def test_get_translator_google_cloud(self) -> None:
        """Test getting Google Cloud translator."""
        translator = get_translator("google_cloud", "test_key")
        assert isinstance(translator, GoogleTranslator)
        assert translator.api_key == "test_key"

    def test_get_translator_unsupported(self) -> None:
        """Test getting unsupported translator."""
        with pytest.raises(TranslationError, match="Unsupported translation service"):
            get_translator("unsupported")

    def test_is_space_language_space_languages(self) -> None:
        """Test space language detection for space languages."""
        # Most languages use spaces
        assert is_space_language("en") is True
        assert is_space_language("es") is True
        assert is_space_language("fr") is True
        assert is_space_language("de") is True

    def test_is_space_language_non_space_languages(self) -> None:
        """Test space language detection for non-space languages."""
        # These should be configured in NON_SPACE_LANGUAGES
        assert is_space_language("zh") is False
        assert is_space_language("ja") is False

    def test_is_space_language_with_region(self) -> None:
        """Test space language detection with region codes."""
        assert is_space_language("en-US") is True
        assert is_space_language("zh-CN") is False

    def test_is_space_language_unknown(self) -> None:
        """Test space language detection for unknown languages."""
        # Unknown languages should default to True
        assert is_space_language("xyz") is True
        assert is_space_language("unknown") is True
