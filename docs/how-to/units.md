---
file_format: mystnb
kernelspec:
  name: python3
  display_name: Python 3
  language: python
---

# Use physical units

This guide shows how to order data that carries physical units, as
[`unxt`](https://unxt.readthedocs.io) Quantities. It needs the `interop` extra
({doc}`../start/install`).

## Pass a unit system

Give each coordinate as a Quantity, and pass a unit system through
`metadata=pcf.StateMetadata(usys=...)`. Every orderer accepts Quantities, and the
result keeps the input units.

```{code-cell} python
import unxt as u

import phasecurvefit as pcf

pos = {"x": u.Q([0.0, 1.0, 2.0], "kpc"), "y": u.Q([0.0, 0.5, 1.0], "kpc")}
vel = {"x": u.Q([1.0, 1.0, 1.0], "km/s"), "y": u.Q([0.5, 0.5, 0.5], "km/s")}
meta = pcf.StateMetadata(usys=u.unitsystems.galactic)

result = pcf.order(pos, vel, metadata=meta)
```

Without the unit system, `pcf.order` raises a `TypeError` asking for one.

## Give scales the right units

Parameters that are lengths or times take a Quantity too. For the local-flow walk,
`metric_scale` has the units its metric expects: a length for the default
{class}`~phasecurvefit.metrics.AlignedMomentumDistanceMetric`, and a time for
{class}`~phasecurvefit.metrics.FullPhaseSpaceDistanceMetric`
({doc}`tune-metric`).

```{code-cell} python
from phasecurvefit.metrics import FullPhaseSpaceDistanceMetric

# A length for the default metric
result = pcf.order(
    pos,
    vel,
    pcf.orderers.LocalFlowOrderer(start_idx=0, metric_scale=u.Q(100.0, "kpc")),
    metadata=meta,
)

# A time for the full phase-space metric
config = pcf.WalkConfig(metric=FullPhaseSpaceDistanceMetric())
result = pcf.order(
    pos,
    vel,
    pcf.orderers.LocalFlowOrderer(
        config=config, start_idx=0, metric_scale=u.Q(1.0, "Gyr")
    ),
    metadata=meta,
)
```

## MST lengths are plain numbers

{class}`~phasecurvefit.orderers.MSTOrderer` works on plain arrays internally. Its
`jump_cap` and `velocity_weight` are plain numbers, read in the length unit of the
unit system, and its `backbone` comes back in the position units.

```{code-cell} python
mst = pcf.orderers.MSTOrderer(k=2, jump_cap=3.0)  # 3 kpc in the galactic system
result = pcf.order(pos, vel, mst, metadata=meta)
result.backbone["x"].unit
```
