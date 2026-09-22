"""Self-Organizing Map ordering, in JAX.

The 1-D SOM of Starkman et al. (2023), re-implemented in JAX. See
:doc:`/guides/som` for the full description.

References
----------
Starkman, N., Bovy, J., Webb, J. J., Calvetti, D., & Somersalo, E. (2023).
*On the Fast Track: Rapid construction of stellar stream paths.*
MNRAS 522(4), 5022-5036. https://arxiv.org/abs/2212.00949

If you use this module, please cite that paper.

"""

__all__: tuple[str, ...] = ("SOM1D", "chord", "densify", "fit", "init_prototypes")

from ._src.som import SOM1D, chord, densify, fit, init_prototypes
