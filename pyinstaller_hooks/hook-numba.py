"""PyInstaller hook: omit optional numba TBB pool extension."""

excludedimports = ["numba.np.ufunc.tbbpool"]
