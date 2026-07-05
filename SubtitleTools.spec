# -*- mode: python ; coding: utf-8 -*-
"""
PyInstaller specification file for SubtitleTools.

Generate ``file_version_info.txt`` before building:
    python scripts/generate_file_version_info.py

This spec file includes optimizations to reduce antivirus false positives:
- Version information resource for legitimacy
- Disabled UPX compression (--noupx) which triggers heuristic detection
- Console application metadata
- Company and product information
"""

from PyInstaller.utils.hooks import collect_all

block_cipher = None

torch_datas, torch_binaries, torch_hiddenimports = collect_all("torch")
whisper_datas, whisper_binaries, whisper_hiddenimports = collect_all("whisper")
jieba_datas, jieba_binaries, jieba_hiddenimports = collect_all("jieba")

hiddenimports = [
    "subtitletools",
    "whisper",
    "torch",
    "jieba",
    "srt",
    "scipy",
    "numpy",
    "tqdm",
    "pyexecjs",
    *torch_hiddenimports,
    *whisper_hiddenimports,
    *jieba_hiddenimports,
]

a = Analysis(
    ["src/subtitletools/__main__.py"],
    pathex=["src"],
    binaries=torch_binaries + whisper_binaries + jieba_binaries,
    datas=[
        ("pyproject.toml", "."),
        *torch_datas,
        *whisper_datas,
        *jieba_datas,
    ],
    hiddenimports=hiddenimports,
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
    upx=False,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=True,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    version="file_version_info.txt",
)
