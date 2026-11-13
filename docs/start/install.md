# Install

phasecurvefit needs Python 3.12 or later. Install it with its optional extras:

::::{tab-set}

:::{tab-item} pip

```bash
pip install "phasecurvefit[all]"
```

:::

:::{tab-item} uv

```bash
uv add phasecurvefit --extra all
```

:::

::::

## Extras

| Extra | Adds | Needed for |
|---|---|---|
| (none) | the core library | ordering and fitting plain JAX arrays |
| `interop` | `unxt` | inputs with physical units ({doc}`../how-to/units`) |
| `kdtree` | `jaxkd` | the {class}`~phasecurvefit.strats.KDTree` query strategy ({doc}`../how-to/kdtree`) |
| `numpyro` | `numpyro` | {func}`~phasecurvefit.sample_track`, a posterior for the track ({doc}`../how-to/track-uncertainty`) |
| `all` | `interop`, `kdtree` and `numpyro` | all optional runtime features |
| `tutorials` | `all`, plus `matplotlib` and `galax` | running the {doc}`tutorial notebooks <../tutorials/index>` |

`all` leaves out `tutorials` on purpose: `all` is for optional features of the
library, while `tutorials` holds packages that only the example notebooks need.
To run the tutorials:

```bash
pip install "phasecurvefit[tutorials]"
```

## Install for a GPU

A plain install ships a CPU-only `jaxlib`. If you have an NVIDIA GPU, JAX prints
this warning and runs on the CPU:

```text
An NVIDIA GPU may be present on this machine, but a CUDA-enabled jaxlib is
not installed. Falling back to cpu.
```

Install JAX's CUDA-enabled build alongside phasecurvefit:

::::{tab-set}

:::{tab-item} pip

```bash
pip install --upgrade "phasecurvefit[all]" "jax[cuda12]"
```

`--upgrade` makes pip replace a CPU-only `jaxlib` that is already installed.

:::

:::{tab-item} uv

```bash
uv add phasecurvefit --extra all
uv add "jax[cuda12]"
```

:::

::::

This pulls in self-contained NVIDIA CUDA and cuDNN wheels, so you don't need the
CUDA toolkit installed system-wide, but you do need a
[compatible NVIDIA driver](https://docs.jax.dev/en/latest/installation.html#nvidia-gpu).
The [JAX installation guide](https://docs.jax.dev/en/latest/installation.html#nvidia-gpu)
covers other CUDA versions, TPUs and ROCm. Your code needs no changes to run on the
GPU; {doc}`../how-to/gpu` shows how to check which device JAX is using.

## Install the development version

To install the latest version from the `main` branch on GitHub:

```bash
uv add git+https://github.com/GalacticDynamics/phasecurvefit.git@main
```

Replace `main` with another branch name to install that branch.

To work on phasecurvefit itself, clone the repository and install it in editable
mode:

```bash
git clone https://github.com/GalacticDynamics/phasecurvefit.git
cd phasecurvefit
uv pip install -e .
```

{doc}`../contributing` describes the development setup and how to run the tests.
