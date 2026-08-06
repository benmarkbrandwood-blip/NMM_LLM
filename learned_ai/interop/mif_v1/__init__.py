"""Independent MIF 1.0 interoperability implementation.

The package implements the frozen MIF wire contract from commit
``7e45d5a3fa970a535ed6a8a8ff5981aba4b9c978``.  It deliberately does not
import the MIF candidate reference runner.
"""

from .adapter import MifInteropAdapter

__all__ = ["MifInteropAdapter"]
