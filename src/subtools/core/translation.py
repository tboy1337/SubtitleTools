"""Translation module for SubtitleTools.

This module provides subtitle translation functionality using various translation services,
adapted from the original SubtranSlate implementation.
"""

import json
import logging
import random
import re
import time
import urllib.error
import urllib.parse
import urllib.request
from abc import ABC, abstractmethod
from typing import Any, Callable, Dict, List, Optional

import execjs
import requests

from ..config.settings import (
    DEFAULT_SRC_LANGUAGE,
    DEFAULT_TARGET_LANGUAGE,
    NON_SPACE_LANGUAGES,
    SUPPORTED_TRANSLATION_SERVICES,
)

logger = logging.getLogger(__name__)


class TranslationError(Exception):
    """Exception raised for translation errors."""


class RateLimitError(TranslationError):
    """Exception raised specifically for rate limiting errors."""


class Translator(ABC):
    """Abstract base class for translation services."""

    @abstractmethod
    def translate(self, text: str, src_lang: str, target_lang: str) -> str:
        """Translate a text from source language to target language."""

    @abstractmethod
    def translate_lines(
        self,
        text_list: List[str],
        src_lang: str,
        target_lang: str,
        progress_callback: Optional[Callable[[int, int, str], None]] = None,
    ) -> str:
        """Translate a list of text lines."""


class TkGenerator:
    """Token generator for Google Translate."""

    def __init__(self) -> None:
        self.js_code = self._get_js_code()

    def _get_js_code(self) -> str:
        """Get JavaScript code for token generation."""
        return """
        var b = function(a, b) {
            for (var d = 0; d < b.length - 2; d += 3) {
                var c = b.charAt(d + 2),
                c = "a" <= c ? c.charCodeAt(0) - 87 : Number(c),
                c = "+" == b.charAt(d + 1) ? a >>> c : a << c;
                a = "+" == b.charAt(d) ? a + c & 4294967295 : a ^ c
            }
            return a
        };
        var tk = function(a, TKK) {
            for (var e = TKK.split("."), h = Number(e[0]) || 0, g = [], d = 0, f = 0; f < a.length; f++) {
                var c = a.charCodeAt(f);
                128 > c ? g[d++] = c : (2048 > c ? g[d++] = c >> 6 | 192 :
                (55296 == (c & 64512) && f + 1 < a.length && 56320 == (a.charCodeAt(f + 1) & 64512) ?
                (c = 65536 + ((c & 1023) << 10) + (a.charCodeAt(++f) & 1023),
                g[d++] = c >> 18 | 240, g[d++] = c >> 12 & 63 | 128) : g[d++] = c >> 12 | 224,
                g[d++] = c >> 6 & 63 | 128), g[d++] = c & 63 | 128)
            }
            a = h;
            for (d = 0; d < g.length; d++) a += g[d], a = b(a, "+-a^+6");
            a = b(a, "+-3^+b+-f");
            a ^= Number(e[1]) || 0;
            0 > a && (a = (a & 2147483647) + 2147483648);
            a %= 1E6;
            return a.toString() + "." + (a ^ h)
        };
        """

    def generate(self, text: str, tkk: str = "0") -> str:
        """Generate token for given text."""
        try:
            ctx = execjs.compile(self.js_code)
            result = ctx.call("tk", text, tkk)
            return str(result) if result is not None else "0"
        except (ValueError, TypeError, AttributeError, Exception) as e:
            logger.warning("Failed to generate token: %s", e)
            return "0"


class GoogleTranslator(Translator):
    """Google Translate implementation."""

    def __init__(self, api_key: Optional[str] = None) -> None:
        self.api_key = api_key
        self.headers = {
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                "(KHTML, like Gecko) Chrome/96.0.4664.110 Safari/537.36"
            )
        }
        self.tk_gen = TkGenerator()
        self.pattern = re.compile(r'\["(.*?)(?:\\n)')
        self.max_limited = 3500

        # Retry configuration
        self.max_retries = 5
        self.initial_backoff = 2
        self.max_backoff = 60
        self.jitter = 0.1

        # Alternative user agents
        self.user_agents = [
            (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                "(KHTML, like Gecko) Chrome/96.0.4664.110 Safari/537.36"
            ),
            (
                "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 "
                "(KHTML, like Gecko) Version/15.1 Safari/605.1.15"
            ),
            (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:94.0) "
                "Gecko/20100101 Firefox/94.0"
            ),
            (
                "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
                "(KHTML, like Gecko) Chrome/96.0.4664.45 Safari/537.36"
            ),
        ]

    def _rotate_user_agent(self) -> None:
        """Rotate the user agent to avoid detection."""
        self.headers["User-Agent"] = random.choice(self.user_agents)

    def _calculate_backoff(self, retry_count: int) -> float:
        """Calculate exponential backoff time with jitter."""
        backoff_time = min(
            self.initial_backoff * (2 ** retry_count),
            self.max_backoff
        )
        jitter = backoff_time * self.jitter * random.random()
        return float(backoff_time + jitter)

    def _make_request_with_retry(
        self,
        url: str,
        data: Optional[str] = None,
        headers: Optional[Dict[str, str]] = None
    ) -> str:
        """Make HTTP request with retry logic for rate limiting."""
        if headers is None:
            headers = self.headers.copy()

        for attempt in range(self.max_retries):
            try:
                if data:
                    req = urllib.request.Request(
                        url, data=data.encode("utf-8"), headers=headers
                    )
                else:
                    req = urllib.request.Request(url, headers=headers)

                with urllib.request.urlopen(req, timeout=30) as response:
                    return str(response.read().decode("utf-8"))

            except urllib.error.HTTPError as e:
                if e.code == 429:  # Too Many Requests
                    if attempt < self.max_retries - 1:
                        backoff_time = self._calculate_backoff(attempt)
                        logger.warning(
                            "Rate limited (attempt %d/%d). Waiting %.2f seconds...",
                            attempt + 1,
                            self.max_retries,
                            backoff_time
                        )
                        time.sleep(backoff_time)
                        self._rotate_user_agent()
                        headers = self.headers.copy()
                        continue
                    raise RateLimitError(
                            f"Rate limit exceeded after {self.max_retries} attempts"
                        ) from e
                # Server errors
                if 500 <= e.code < 600:
                    if attempt < self.max_retries - 1:
                        backoff_time = self._calculate_backoff(attempt)
                        logger.warning(
                            "Server error %d (attempt %d/%d). Waiting %.2f seconds...",
                            e.code,
                            attempt + 1,
                            self.max_retries,
                            backoff_time
                        )
                        time.sleep(backoff_time)
                        continue
                    raise TranslationError(
                        f"Server error {e.code} after {self.max_retries} attempts"
                    ) from e

                raise TranslationError(f"HTTP error {e.code}: {e.reason}") from e

            except (urllib.error.URLError, Exception) as e:
                if attempt < self.max_retries - 1:
                    backoff_time = self._calculate_backoff(attempt)
                    logger.warning(
                        "Request failed (attempt %d/%d): %s. Waiting %.2f seconds...",
                        attempt + 1,
                        self.max_retries,
                        e,
                        backoff_time
                    )
                    time.sleep(backoff_time)
                    continue
                raise TranslationError(
                    f"Request failed after {self.max_retries} attempts: {e}"
                ) from e

        raise TranslationError("Maximum retry attempts exceeded")

    def translate(self, text: str, src_lang: str, target_lang: str) -> str:
        """Translate text using Google Translate."""
        if not text.strip():
            return text

        # Use cloud API if available
        if self.api_key:
            return self._translate_with_api(text, src_lang, target_lang)

        # Use web interface
        return self._translate_with_web(text, src_lang, target_lang)

    def _translate_with_api(self, text: str, src_lang: str, target_lang: str) -> str:
        """Translate using Google Cloud Translation API."""
        try:
            url = f"https://translation.googleapis.com/language/translate/v2?key={self.api_key}"

            data = {
                "q": text,
                "source": src_lang,
                "target": target_lang,
                "format": "text"
            }

            response = requests.post(url, json=data, timeout=30)
            response.raise_for_status()

            result = response.json()

            if "data" in result and "translations" in result["data"]:
                translated_text = result["data"]["translations"][0]["translatedText"]
                return str(translated_text)
            raise TranslationError("Invalid API response format")

        except requests.RequestException as e:
            raise TranslationError(f"API request failed: {e}") from e

    def _translate_with_web(self, text: str, src_lang: str, target_lang: str) -> str:
        """Translate using Google Translate web interface."""
        if len(text) > self.max_limited:
            raise TranslationError(
                f"Text too long for web interface: {len(text)} > {self.max_limited}"
            )

        try:
            # Generate token
            tk = self.tk_gen.generate(text)

            # Build request URL
            params = {
                "client": "gtx",
                "sl": src_lang,
                "tl": target_lang,
                "hl": target_lang,
                "dt": "t",
                "ie": "UTF-8",
                "oe": "UTF-8",
                "otf": "1",
                "ssel": "0",
                "tsel": "0",
                "kc": "7",
                "q": text,
                "tk": tk,
            }

            url = "https://translate.googleapis.com/translate_a/single?" + urllib.parse.urlencode(params)

            # Make request
            response = self._make_request_with_retry(url)

            # Parse response
            response_data = json.loads(response)

            if not response_data or not response_data[0]:
                raise TranslationError("Empty translation response")

            # Extract translated text
            translated_parts = []
            for part in response_data[0]:
                if part[0]:
                    translated_parts.append(part[0])

            # Join parts - use newline if original text contained newlines
            if "\n" in text:
                return "\n".join(translated_parts)
            return "".join(translated_parts)

        except (json.JSONDecodeError, IndexError, KeyError) as e:
            raise TranslationError(f"Failed to parse translation response: {e}") from e

    def translate_lines(
        self,
        text_list: List[str],
        src_lang: str,
        target_lang: str,
        progress_callback: Optional[Callable[[int, int, str], None]] = None,
    ) -> str:
        """Translate a list of text lines."""
        if not text_list:
            return ""

        combined_text = "\n".join(text_list)

        # For very long text, translate in chunks
        if len(combined_text) > self.max_limited:
            return self._translate_in_chunks(
                text_list, src_lang, target_lang, progress_callback
            )

        # Single translation
        if progress_callback:
            progress_callback(0, len(text_list), "Translating...")

        translated = self.translate(combined_text, src_lang, target_lang)

        if progress_callback:
            progress_callback(len(text_list), len(text_list), "Complete")

        return translated

    def _translate_in_chunks(
        self,
        text_list: List[str],
        src_lang: str,
        target_lang: str,
        progress_callback: Optional[Callable[[int, int, str], None]] = None,
    ) -> str:
        """Translate text in chunks to handle length limits."""
        chunks: List[List[str]] = []
        current_chunk: List[str] = []
        current_length = 0

        # Split into chunks
        for line in text_list:
            line_length = len(line) + 1  # +1 for newline

            if current_length + line_length > self.max_limited and current_chunk:
                chunks.append(current_chunk)
                current_chunk = [line]
                current_length = line_length
            else:
                current_chunk.append(line)
                current_length += line_length

        if current_chunk:
            chunks.append(current_chunk)

        # Translate chunks
        translated_chunks = []
        processed_lines = 0

        for i, chunk in enumerate(chunks):
            if progress_callback:
                progress_callback(
                    processed_lines,
                    len(text_list),
                    f"Translating chunk {i + 1}/{len(chunks)}"
                )

            chunk_text = "\n".join(chunk)
            translated_chunk = self.translate(chunk_text, src_lang, target_lang)
            translated_chunks.append(translated_chunk)

            processed_lines += len(chunk)

            # Add delay between chunks to avoid rate limiting
            if i < len(chunks) - 1:
                time.sleep(1)

        if progress_callback:
            progress_callback(len(text_list), len(text_list), "Complete")

        return "\n".join(translated_chunks)


class SubtitleTranslator:
    """Main subtitle translation class."""

    def __init__(
        self,
        translation_service: str = "google",
        api_key: Optional[str] = None,
    ):
        """Initialize the subtitle translator.

        Args:
            translation_service: Translation service to use
            api_key: API key for the translation service

        Raises:
            TranslationError: If service is not supported
        """
        if translation_service not in SUPPORTED_TRANSLATION_SERVICES:
            raise TranslationError(
                f"Unsupported translation service: {translation_service}"
            )

        self.service_name = translation_service
        self.translator = self._get_translator(translation_service, api_key)

        logger.info("Initialized translator with service: %s", translation_service)

    def _get_translator(self, service: str, api_key: Optional[str]) -> Translator:
        """Get translator instance for the specified service."""
        if service in ["google", "google_cloud"]:
            return GoogleTranslator(api_key)
        raise TranslationError(f"Unsupported translation service: {service}")

    def translate_text(
        self,
        text: str,
        src_lang: str = DEFAULT_SRC_LANGUAGE,
        target_lang: str = DEFAULT_TARGET_LANGUAGE,
        progress_callback: Optional[Callable[[int, int, str], None]] = None,
    ) -> str:
        """Translate a single text string.

        Args:
            text: Text to translate
            src_lang: Source language code
            target_lang: Target language code
            progress_callback: Optional progress callback

        Returns:
            Translated text

        Raises:
            TranslationError: If translation fails
        """
        if not text.strip():
            return text

        try:
            if progress_callback:
                progress_callback(0, 1, "Translating...")

            result = self.translator.translate(text, src_lang, target_lang)

            if progress_callback:
                progress_callback(1, 1, "Complete")

            return result

        except Exception as e:
            raise TranslationError(f"Translation failed: {e}") from e

    def translate_lines(
        self,
        lines: List[str],
        src_lang: str = DEFAULT_SRC_LANGUAGE,
        target_lang: str = DEFAULT_TARGET_LANGUAGE,
        progress_callback: Optional[Callable[[int, int, str], None]] = None,
    ) -> List[str]:
        """Translate a list of text lines.

        Args:
            lines: List of text lines to translate
            src_lang: Source language code
            target_lang: Target language code
            progress_callback: Optional progress callback

        Returns:
            List of translated lines

        Raises:
            TranslationError: If translation fails
        """
        if not lines:
            return []

        try:
            # Filter out empty lines but remember their positions
            non_empty_lines: List[str] = []
            line_mapping: Dict[int, int] = {}

            for i, line in enumerate(lines):
                if line.strip():
                    line_mapping[len(non_empty_lines)] = i
                    non_empty_lines.append(line)

            if not non_empty_lines:
                return lines

            # Translate non-empty lines
            translated_text = self.translator.translate_lines(
                non_empty_lines, src_lang, target_lang, progress_callback
            )

            translated_lines = translated_text.split("\n")

            # Reconstruct original structure
            result = lines.copy()
            for j, translated_line in enumerate(translated_lines):
                if j in line_mapping:
                    original_index = line_mapping[j]
                    result[original_index] = translated_line

            return result

        except Exception as e:
            raise TranslationError(f"Batch translation failed: {e}") from e

    def get_service_info(self) -> Dict[str, Any]:
        """Get information about the translation service.

        Returns:
            Dictionary with service information
        """
        return {
            "service": self.service_name,
            "has_api_key": hasattr(self.translator, "api_key") and self.translator.api_key is not None,
            "max_text_length": getattr(self.translator, "max_limited", None),
        }


def get_translator(service: str, api_key: Optional[str] = None) -> Translator:
    """Factory function to get a translator instance.

    Args:
        service: Translation service name
        api_key: Optional API key

    Returns:
        Translator instance

    Raises:
        TranslationError: If service is not supported
    """
    if service in ["google", "google_cloud"]:
        return GoogleTranslator(api_key)
    raise TranslationError(f"Unsupported translation service: {service}")


def is_space_language(language_code: str) -> bool:
    """Check if a language uses spaces between words.

    Args:
        language_code: Language code to check

    Returns:
        True if language uses spaces between words, defaults to True for unknown languages
    """
    # Get base language code without region
    base_lang = language_code.split("-")[0].lower()

    # Check if explicitly listed as non-space language
    if base_lang in [lang.lower() for lang in NON_SPACE_LANGUAGES]:
        return False

    # Default to True (most languages use spaces, including unknown ones)
    return True
