# SubtitleTools

A tool for subtitle processing workflows, including extraction, conversion and optimization.

## Documentation

- [Installation](docs/installation.md)
- [Usage](docs/usage.md)
- [Translation services](docs/translation.md)
- [Troubleshooting](docs/troubleshooting.md)
- [Development](docs/development.md)

## Features

SubtitleTools provides a complete subtitle processing pipeline:

### 🎬 Transcription
- **Audio/Video to Subtitles**: Generate subtitles from video/audio files using OpenAI's Whisper
- **Multiple Model Sizes**: Support for tiny, base, small, medium, and large Whisper models
- **Batch Processing**: Process entire directories of video files
- **Audio Extraction**: Extract audio from various video formats using FFmpeg
- **Segment Control**: Control subtitle segment length and timing
- **Multi-language Support**: Transcribe in 100+ languages with automatic detection

### 🌐 Translation
- **Language Translation**: Translate subtitle files between 50+ languages
- **Rate Limiting Protection**: Robust handling of API rate limits with retry backoff
- **Translation Services**: `google` (web, requires Node.js) or `google_cloud` (API key)
- **Batch Translation**: Translate multiple subtitle files at once

### 🔤 Encoding Conversion
- **Multiple Encodings**: Support for 28+ character encodings
- **Language-Specific Recommendations**: Smart encoding suggestions based on language
- **Batch Encoding**: Convert multiple files to various encodings
- **Auto-Detection**: Automatic source encoding detection

### ⚙️ Post-Processing
- **Native Processing**: Built-in subtitle post-processing without external dependencies
- **Common Fixes**: Apply common subtitle error corrections
- **Format Conversion**: Convert between SRT, ASS, VTT, and other formats
- **Line Splitting**: Automatically split long subtitle lines
- **OCR Fixes**: Correct common OCR errors
- **Hearing Impaired Removal**: Remove hearing impaired text markers

### 🔄 Workflows
- **End-to-End Processing**: Video → Subtitles → Translation → Post-processing
- **Flexible Workflows**: Mix and match operations as needed
- **Resume Capability**: Resume interrupted operations
- **Comprehensive Logging**: Detailed logging for troubleshooting

## 📦 Installation

### Prerequisites

1. **Python 3.12+**
2. **FFmpeg** (for video/audio processing)
3. **Node.js** (for `google` web translation via `pyexecjs`; optional if using `google_cloud` with an API key)

See [docs/installation.md](docs/installation.md) for details.

### Installing FFmpeg

#### Windows
- Download the latest static essentials build from [gyan.dev](https://www.gyan.dev/ffmpeg/builds/).
- Use a package manager:
```cmd
# Using Winget
winget install Gyan.FFmpeg.Essentials
```

```cmd
# Using Chocolatey  
choco install ffmpeg
```

#### macOS
```bash
brew install ffmpeg
```

#### Linux
```bash
# Ubuntu/Debian
sudo apt update && sudo apt install ffmpeg

# Fedora
sudo dnf install ffmpeg
```

### Installing SubtitleTools

#### Using pip (Recommended)

The easiest way to install SubtitleTools is directly from PyPI:

```bash
pip install subtitletools
```

That's it! The `subtitletools` command will be available in your terminal.

#### From Source

Alternatively, you can install from source for development or to get the latest unreleased features:

```bash
git clone https://github.com/tboy1337/SubtitleTools.git
cd SubtitleTools

# Install with development dependencies
pip install -e ".[dev]"
```

### Releases

Pre-built Windows executables are published on [GitHub Releases](https://github.com/tboy1337/SubtitleTools/releases) when a version tag (e.g. `v1.0.3`) is pushed. pip installs remain the recommended cross-platform option.

#### Using the Tool

```bash
python -m subtitletools --help
subtitletools --help
```

## 🚀 Quick Start

### Generate Subtitles from Video
```bash
# Basic transcription
python -m subtitletools transcribe video.mp4

# With specific model and language
python -m subtitletools transcribe video.mp4 --model medium --language en

# Batch process directory
python -m subtitletools transcribe videos/ --batch --output subtitles/
```

### Translate Existing Subtitles
```bash
# Translate English to Spanish
python -m subtitletools translate input.srt output.srt --src-lang en --target-lang es

# Batch translate directory
python -m subtitletools translate subtitles/ translated/ --batch --src-lang en --target-lang fr
```

### Convert Encoding
```bash
# Convert to specific encoding
python -m subtitletools encode input.srt --to-encoding utf-8

# Convert to recommended encodings for Thai
python -m subtitletools encode thai_subtitle.srt --recommended --language th
```

### Complete Workflow
```bash
# Generate and translate subtitles in one go
python -m subtitletools workflow video.mp4 --target-lang es --model small

# With post-processing
python -m subtitletools workflow video.mp4 --target-lang fr --fix-common-errors --remove-hi
```

## Subtitle Post-Processing

SubtitleTools includes built-in subtitle post-processing functionality with no external dependencies required.

```bash
# Fix common errors
python -m subtitletools workflow video.mp4 --fix-common-errors

# Remove text for hearing impaired
python -m subtitletools workflow video.mp4 --remove-hi

# Apply multiple fixes at once
python -m subtitletools workflow video.mp4 --fix-common-errors --remove-hi --auto-split-long-lines
```

Available post-processing options:
- `--fix-common-errors`: Fix common subtitle issues (overlapping times, short/long display times, spacing, etc.)
- `--remove-hi`: Remove hearing impaired text (content in brackets, parentheses, speaker names, etc.)
- `--auto-split-long-lines`: Split long subtitle lines intelligently
- `--fix-punctuation`: Fix punctuation issues (ellipsis, quotation marks, multiple punctuation, etc.)
- `--ocr-fix`: Apply OCR error corrections (common character misrecognitions)
- `--convert-to`: Convert format (srt, ass, ssa, vtt, sami)

All post-processing is performed using native Python implementations for maximum compatibility and performance.

## 🛠️ Command Reference

### Core Commands

- `transcribe` - Generate subtitles from video/audio
- `translate` - Translate subtitle files between languages  
- `encode` - Convert subtitle file encodings
- `workflow` - Run end-to-end subtitle workflows

### Transcription Options
- `--model` - Whisper model size (tiny, base, small, medium, large)
- `--language` - Source language code for transcription
- `--max-segment-length` - Maximum characters per subtitle segment
- `--batch` - Process entire directories

### Translation Options  
- `--src-lang` - Source language code
- `--target-lang` - Target language code
- `--service` - Translation service: `google` (web) or `google_cloud` (requires `--api-key`)
- `--api-key` - Translation service API key
- `--both` - Keep both original and translated text

### Post-Processing Options
- `--fix-common-errors` - Apply common subtitle fixes
- `--remove-hi` - Remove hearing impaired text
- `--auto-split-long-lines` - Split long lines automatically
- `--fix-punctuation` - Fix punctuation issues
- `--ocr-fix` - Apply OCR error corrections
- `--convert-to` - Convert to different format (srt, ass, ssa, vtt, sami)

## 🌐 Supported Languages

SubtitleTools supports 100+ languages for transcription and 50+ for translation, including:

| Language | Transcription | Translation | Code |
|----------|---------------|-------------|------|
| English | ✅ | ✅ | en |
| Spanish | ✅ | ✅ | es |  
| French | ✅ | ✅ | fr |
| German | ✅ | ✅ | de |
| Chinese (Simplified) | ✅ | ✅ | zh-CN |
| Japanese | ✅ | ✅ | ja |
| Korean | ✅ | ✅ | ko |
| Russian | ✅ | ✅ | ru |
| Arabic | ✅ | ✅ | ar |
| Thai | ✅ | ✅ | th |

*For a complete list of supported languages, check the Whisper documentation for transcription and Google Translate documentation for translation support.*

## 🔧 Configuration

Configuration is handled through command-line arguments. The tool automatically creates necessary directories in your system's application data folder (e.g., `~/.subtitletools/` on Unix-like systems or `%APPDATA%/SubtitleTools/` on Windows) for caching and temporary files.

## 🧪 Development

Install the package with development dependencies:

```bash
pip install -e ".[dev]"
```

Alternatively, install runtime and dev dependencies separately:

```bash
pip install -e .
pip install -r requirements-dev.txt
```

Run the local verification script (formatting, type checks, lint, security scan, tests):

```bash
py scripts/verify.py
py scripts/verify.py --fix
```

### Running Tests

```bash
pytest
pytest --cov=src
pytest -m unit
pytest -m integration
```

Tests enforce coverage reporting (see `pytest.ini` and `.coveragerc`). `py scripts/verify.py` runs the full local quality gate before release.

## 📊 Performance Tips

### Transcription
- Transcription runs on CPU (no GPU required or supported)
- Start with smaller models for testing
- Use batch processing for multiple files
- Consider splitting very large files

### Translation
- Use API keys for better rate limits
- Enable resume functionality for large jobs
- Process during off-peak hours

### Post-Processing  
- No external dependencies required
- Native Python implementation for fast processing
- Use batch processing for multiple files

## ⚠️ Requirements

### Required Dependencies
- openai-whisper (transcription)
- torch (ML processing)
- scipy, numpy (audio processing) 
- tqdm (progress bars)
- pyexecjs (translation engine)
- srt (subtitle parsing)
- requests (API communication)
- jieba (Chinese text segmentation)

## 📄 License

CRL License - see [LICENSE.md](LICENSE.md) file for details.
