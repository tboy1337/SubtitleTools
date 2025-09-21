"""Configuration settings for SubtitleTools."""

import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

# Default configuration values
DEFAULT_WHISPER_MODEL = "small"
DEFAULT_ENCODING = "UTF-8"
DEFAULT_TIMEOUT = 300  # 5 minutes
DEFAULT_SRC_LANGUAGE = "en"
DEFAULT_TARGET_LANGUAGE = "zh-CN"
DEFAULT_TRANSLATION_MODE = "split"
DEFAULT_MAX_SEGMENT_LENGTH: Optional[int] = None

# Supported file extensions
SUPPORTED_VIDEO_EXTENSIONS = [
    "mp4", "mkv", "avi", "mov", "webm", "flv", "wmv", "m4v", "mpg", "mpeg"
]

SUPPORTED_AUDIO_EXTENSIONS = [
    "wav", "mp3", "aac", "flac", "ogg", "m4a", "wma"
]

SUPPORTED_SUBTITLE_FORMATS = [
    "srt", "vtt", "ass", "ssa", "sub", "sbv", "ttml"
]

# Translation service configurations
SUPPORTED_TRANSLATION_SERVICES = [
    "google", "google_cloud"
]

# Whisper model configurations
WHISPER_MODELS = [
    "tiny", "base", "small", "medium", "large"
]

# Encoding configurations
SUPPORTED_ENCODINGS = [
    # UTF encodings
    "utf-8", "utf-8-sig", "utf-16", "utf-16-le", "utf-16-be", "utf-32",
    # Western European
    "iso-8859-1", "iso-8859-15", "cp1252", "cp850",
    # Central European
    "iso-8859-2", "cp1250", "cp852",
    # Cyrillic
    "iso-8859-5", "cp1251", "koi8-r",
    # Greek
    "iso-8859-7", "cp1253",
    # Turkish
    "iso-8859-9", "cp1254",
    # Hebrew
    "iso-8859-8", "cp1255",
    # Arabic
    "iso-8859-6", "cp1256",
    # Thai
    "tis-620", "cp874", "iso-8859-11",
    # Asian
    "gb2312", "cp936", "big5", "cp950",  # Chinese
    "shift-jis", "euc-jp", "cp932",      # Japanese
    "euc-kr", "cp949",                   # Korean
    # Vietnamese
    "cp1258",
]

# Language-specific encoding recommendations
LANGUAGE_ENCODINGS: Dict[str, List[str]] = {
    "en": ["utf-8", "utf-8-sig", "cp1252", "iso-8859-1"],
    "fr": ["utf-8", "utf-8-sig", "cp1252", "iso-8859-1", "iso-8859-15"],
    "de": ["utf-8", "utf-8-sig", "cp1252", "iso-8859-1", "iso-8859-15"],
    "es": ["utf-8", "utf-8-sig", "cp1252", "iso-8859-1", "iso-8859-15"],
    "it": ["utf-8", "utf-8-sig", "cp1252", "iso-8859-1", "iso-8859-15"],
    "pt": ["utf-8", "utf-8-sig", "cp1252", "iso-8859-1", "iso-8859-15"],
    "ru": ["utf-8", "utf-8-sig", "cp1251", "koi8-r", "iso-8859-5"],
    "zh": ["utf-8", "utf-8-sig", "gb2312", "cp936"],
    "zh-CN": ["utf-8", "utf-8-sig", "gb2312", "cp936"],
    "zh-TW": ["utf-8", "utf-8-sig", "big5", "cp950"],
    "ja": ["utf-8", "utf-8-sig", "shift-jis", "euc-jp", "cp932"],
    "ko": ["utf-8", "utf-8-sig", "euc-kr", "cp949"],
    "th": ["utf-8", "utf-8-sig", "tis-620", "cp874", "iso-8859-11"],
    "ar": ["utf-8", "utf-8-sig", "cp1256", "iso-8859-6"],
    "he": ["utf-8", "utf-8-sig", "cp1255", "iso-8859-8"],
    "tr": ["utf-8", "utf-8-sig", "cp1254", "iso-8859-9"],
    "el": ["utf-8", "utf-8-sig", "cp1253", "iso-8859-7"],
    "vi": ["utf-8", "utf-8-sig", "cp1258"],
}

# Languages that use spaces between words
SPACE_LANGUAGES = [
    "en", "fr", "de", "es", "it", "pt", "ru", "ar", "he", "tr", "el", "vi"
]

# Languages that typically do not use spaces between words
NON_SPACE_LANGUAGES = [
    "zh", "zh-cn", "zh-tw", "ja", "ko", "th"
]

# Application directories
def get_app_data_dir() -> Path:
    """Get the application data directory."""
    if os.name == "nt":  # Windows
        app_data = os.environ.get("APPDATA", os.path.expanduser("~"))
        return Path(app_data) / "SubtitleTools"
    # Unix-like
    return Path.home() / ".subtitletools"

def get_cache_dir() -> Path:
    """Get the cache directory."""
    return get_app_data_dir() / "cache"

def get_temp_dir() -> Path:
    """Get the temporary files directory."""
    return get_app_data_dir() / "temp"

def get_logs_dir() -> Path:
    """Get the logs directory."""
    return get_app_data_dir() / "logs"

# Global configuration storage
_config: Dict[str, Union[str, int, float, bool]] = {}

def get_config(key: str, default: Optional[Union[str, int, float, bool]] = None) -> Optional[Union[str, int, float, bool]]:
    """Get a configuration value.

    Args:
        key: Configuration key
        default: Default value if key not found

    Returns:
        Configuration value or default
    """
    return _config.get(key, default)

def set_config(key: str, value: Union[str, int, float, bool]) -> None:
    """Set a configuration value.

    Args:
        key: Configuration key
        value: Configuration value
    """
    _config[key] = value

def get_all_config() -> Dict[str, Any]:
    """Get all configuration values.

    Returns:
        Dictionary of all configuration values
    """
    return _config.copy()

def reset_config() -> None:
    """Reset configuration to empty state."""
    _config.clear()

# Initialize directories on import
def _ensure_directories() -> None:
    """Ensure application directories exist."""
    for dir_func in [get_app_data_dir, get_cache_dir, get_temp_dir, get_logs_dir]:
        directory = dir_func()
        directory.mkdir(parents=True, exist_ok=True)

# Auto-initialize directories
_ensure_directories()
