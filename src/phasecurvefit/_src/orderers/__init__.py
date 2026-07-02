"""Internal orderers package.

Kept import-light on purpose: importing ``phasecurvefit._src.orderers.result``
(which ``algorithm`` depends on) must not pull in ``localflow`` / ``base``,
which in turn import ``algorithm``. The public surface is assembled by the
top-level ``phasecurvefit.orderers`` shim.
"""

__all__: tuple[str, ...] = ()
