"""Independent MIF 1.0 interoperability implementation.

The package implements the frozen MIF wire contract from commit
``f37ddfeb5fb8479991fa38eeb03c797bef8ae408``.  It deliberately does not
import the MIF candidate reference runner.
"""

from .adapter import MifInteropAdapter

__all__ = ["MifInteropAdapter"]
