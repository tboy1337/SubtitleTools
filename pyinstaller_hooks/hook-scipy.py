"""PyInstaller hook: scipy without removed private modules."""

excludedimports = ["scipy.special._cdflib"]
