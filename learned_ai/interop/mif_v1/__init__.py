"""Independent MIF 1.0 interoperability implementation.

The package implements the frozen MIF wire contract from commit
``7e45d5a3fa970a535ed6a8a8ff5981aba4b9c978`` and is pinned to Suite
candidate commit ``3ee7e57c7d4c7208be91f62914f344a587fb0f70``.  It deliberately
does not import the MIF candidate reference runner.
"""

from .adapter import MifInteropAdapter

__all__ = ["MifInteropAdapter"]
