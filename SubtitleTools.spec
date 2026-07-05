# -*- mode: python ; coding: utf-8 -*-
# PyInstaller specification for SubtitleTools Windows executable.

from pathlib import Path

block_cipher = None
root = Path(SPECPATH)

a = Analysis(
    [str(root / "scripts" / "pyi_entry.py")],
    pathex=[str(root / "src")],
    binaries=[],
    datas=[(str(root / "pyproject.toml"), ".")],
    hiddenimports=[
        "subtitletools",
        "subtitletools.__main__",
        "subtitletools._version",
        "subtitletools.cli",
        "subtitletools.config",
        "subtitletools.config.settings",
        "subtitletools.core",
        "subtitletools.core.subtitle",
        "subtitletools.core.transcription",
        "subtitletools.core.translation",
        "subtitletools.core.workflow",
        "subtitletools.utils",
        "subtitletools.utils.audio",
        "subtitletools.utils.common",
        "subtitletools.utils.encoding",
        "subtitletools.utils.format_converter",
        "subtitletools.utils.postprocess",
        "subtitletools.utils.subtitle_fixes",
        "whisper",
        "torch",
        "scipy",
        "numpy",
        "tqdm",
        "execjs",
        "srt",
        "requests",
        "jieba",
    ],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.zipfiles,
    a.datas,
    [],
    name="SubtitleTools",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=True,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    version=str(root / "file_version_info.txt"),
)
