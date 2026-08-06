"""Independent MIF 1.0 interoperability implementation.

The package implements the frozen MIF wire contract from commit
``0693353fe0821dcbbf547cc1eb9b679dcf2f90b8``.  It deliberately does not
import the MIF candidate reference runner.
"""

from .adapter import MifInteropAdapter

__all__ = ["MifInteropAdapter"]
