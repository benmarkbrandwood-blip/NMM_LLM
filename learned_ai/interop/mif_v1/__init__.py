"""Independent MIF 1.0 interoperability implementation.

The package implements the frozen MIF wire contract from commit
``83e4b758f624f3059c7ba289d4d4429eed0a710a``.  It deliberately does not
import the MIF candidate reference runner.
"""

from .adapter import MifInteropAdapter

__all__ = ["MifInteropAdapter"]
