# Run on a GPU

This guide shows how to check that phasecurvefit runs on your GPU.

First install JAX's CUDA-enabled build ({doc}`../start/install`). A plain install
runs on the CPU and warns that a CUDA-enabled `jaxlib` is not installed.

Your code needs no changes. JAX places arrays and computations on the first
available accelerator:

```python
import jax
import jax.numpy as jnp
import phasecurvefit as pcf

print(jax.devices())  # e.g. [CudaDevice(id=0)] on a GPU, [CpuDevice(id=0)] otherwise

position = {"x": jnp.array([0.0, 1.0, 2.0, 3.0])}
velocity = {"x": jnp.array([1.0, 1.0, 1.0, 1.0])}
result = pcf.order(
    position, velocity, pcf.orderers.LocalFlowOrderer(start_idx=0, metric_scale=1.0)
)
print(result.indices.devices())
```

The MST's SciPy neighbour search runs on the host. Under `jit` the MST switches to
the JAX kd-tree, which runs on the device ({doc}`jax`).

The [JAX installation guide](https://docs.jax.dev/en/latest/installation.html)
covers other CUDA versions, TPUs and ROCm.
