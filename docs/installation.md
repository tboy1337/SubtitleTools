# Installation

## Requirements

- Python 3.12 or newer
- [FFmpeg](https://ffmpeg.org/) on your PATH (required for video/audio extraction)
- Node.js (recommended for `google` web translation via `pyexecjs`)

## Install from PyPI

```bash
pip install subtitletools
```

## Install from source

```bash
git clone https://github.com/tboy1337/SubtitleTools.git
cd SubtitleTools
pip install -e ".[dev]"
```

## Windows executable

Tagged releases publish a Windows `.exe` on [GitHub Releases](https://github.com/tboy1337/SubtitleTools/releases). pip remains the recommended cross-platform install.

## Verify installation

```bash
subtitletools --version
ffmpeg -version
```

For web translation without a Google Cloud API key, verify Node.js:

```bash
node --version
```
