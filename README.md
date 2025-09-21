# SubtitleTools

A tool for subtitle processing workflows, including extraction, conversion and optimization.

## 🚀 Features

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
- **Context-Aware Translation**: Smart translation modes for better accuracy
- **Rate Limiting Protection**: Robust handling of API rate limits with resume capability
- **Multiple Translation Services**: Support for Google Translate and Google Cloud Translation API
- **Batch Translation**: Translate multiple subtitle files at once

### 🔤 Encoding Conversion
- **Multiple Encodings**: Support for 40+ character encodings
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

1. **Python 3.8+**
2. **FFmpeg** (for video/audio processing)

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

#### From Source
```bash
git clone https://github.com/tboy1337/SubtitleTools.git
cd SubtitleTools
pip install -r requirements.txt

# For development
pip install -r requirements-dev.txt
```

#### Using the Tool
```bash
# Direct execution
python run.py --help

# After installation  
python -m subtitletools --help
```

## 🚀 Quick Start

### Generate Subtitles from Video
```bash
# Basic transcription
python run.py transcribe video.mp4

# With specific model and language
python run.py transcribe video.mp4 --model medium --language en

# Batch process directory
python run.py transcribe videos/ --batch --output subtitles/
```

### Translate Existing Subtitles
```bash
# Translate English to Spanish
python run.py translate input.srt output.srt --src-lang en --target-lang es

# Batch translate directory
python run.py translate subtitles/ translated/ --batch --src-lang en --target-lang fr
```

### Convert Encoding
```bash
# Convert to specific encoding
python run.py encode input.srt --to-encoding utf-8

# Convert to recommended encodings for Thai
python run.py encode thai_subtitle.srt --recommended --language th
```

### Complete Workflow
```bash
# Generate and translate subtitles in one go
python run.py workflow video.mp4 --target-lang es --model small

# With post-processing
python run.py workflow video.mp4 --target-lang fr --fix-common-errors --remove-hi
```

## Subtitle Post-Processing

SubtitleTools includes built-in subtitle post-processing functionality with no external dependencies required.

```bash
# Fix common errors
python run.py workflow video.mp4 --fix-common-errors

# Remove text for hearing impaired
python run.py workflow video.mp4 --remove-hi

# Apply multiple fixes at once
python run.py workflow video.mp4 --fix-common-errors --remove-hi --auto-split-long-lines
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
- `--mode` - Translation mode (naive, split)
- `--api-key` - Translation service API key
- `--both` - Keep both original and translated text

### Post-Processing Options
- `--fix-common-errors` - Apply common subtitle fixes
- `--remove-hi` - Remove hearing impaired text
- `--auto-split-long-lines` - Split long lines automatically
- `--fix-punctuation` - Fix punctuation issues
- `--ocr-fix` - Apply OCR error corrections
- `--convert-to` - Convert to different format (srt, ass, vtt)

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

[Complete language list](docs/languages.md)

## 🔧 Configuration

### Environment Variables
- `SUBTITLETOOLS_API_KEY` - Default translation API key
- `SUBTITLETOOLS_LOG_LEVEL` - Logging level (DEBUG, INFO, WARNING, ERROR)
- `SUBTITLETOOLS_TEMP_DIR` - Custom temporary directory

### Configuration Files
- `~/.subtitletools/config.yaml` - User configuration
- `subtitletools.yaml` - Project-specific configuration

## 🧪 Development

### Running Tests
```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=src

# Run specific test categories
pytest -m unit
pytest -m integration
```

### Code Quality
```bash
# Format code
black src/
isort src/

# Linting
pylint src/
mypy src/
```

## 📊 Performance Tips

### Transcription
- Use GPU acceleration when available
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
- pyexecjs (translation engine)
- srt (subtitle parsing)
- requests (API communication)

### Optional Dependencies
- jieba (Chinese text segmentation)

## 📄 License

CRL License - see [LICENSE.md](LICENSE.md) file for details.
