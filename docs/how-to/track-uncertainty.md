---
file_format: mystnb
kernelspec:
  name: python3
  display_name: Python 3
  language: python
---

# Put an uncertainty band on the track

This guide shows two ways to draw a band whose width is the track's uncertainty: fits
to bootstrap resamples of the data, and draws from the posterior of a probabilistic
model of the track. {doc}`../explanation/track-uncertainty` explains what each band
includes and how well it is calibrated.

## Fit the ensemble

{func}`~phasecurvefit.bootstrap_track` takes the same data and options as
{func}`~phasecurvefit.fit_track`. It fits the data once, then refits `n_samples`
resamples of it, each drawn with replacement. Each resample costs one
{func}`~phasecurvefit.fit_track`. The epoch counts and `n_samples` here are small to
keep the example fast. On real data, keep the default epochs and use at least 20
resamples, since the band's 95% level comes from the tails of the members.

```{code-cell} python
import jax
import jax.numpy as jnp
import numpy as np
import phasecurvefit as pcf

rng = np.random.default_rng(0)
ang = jnp.linspace(0.0, jnp.pi, 200)
pos = {
    "x": 5 * jnp.cos(ang) + rng.normal(0, 0.2, 200),
    "y": 5 * jnp.sin(ang) + rng.normal(0, 0.2, 200),
}
vel = {"x": -jnp.sin(ang), "y": jnp.cos(ang)}

config = pcf.nn.TrainingConfig(
    n_epochs_encoder=100, n_epochs_decoder=50, n_epochs_both=50, show_pbar=False
)
ensemble = pcf.bootstrap_track(
    pos, vel, key=jax.random.key(0), n_samples=5, training_config=config
)
len(ensemble.members)
```

`ensemble.reference` is the fit to all the data, an
{class}`~phasecurvefit.nn.AutoencoderResult` like the one `fit_track` returns.
`ensemble.members` are the fits to the resamples.

## Draw the band

{meth}`~phasecurvefit.TrackEnsemble.radius` gives, at points on the reference track,
the radius of the tube about it that holds a given fraction of the members. Each fit
has its own $\gamma$, so the members are compared with the reference in space, across
the track, and not at equal $\gamma$.

```{code-cell} python
import matplotlib.pyplot as plt

ref = ensemble.reference
gamma = jnp.linspace(*ref.track_range, 200)
track = ref(gamma)
r68 = ensemble.radius(gamma, 0.68)

# the normal to the track, to draw the radius on either side of it
tangent = jnp.stack(jnp.gradient(jnp.stack([track["x"], track["y"]]), axis=1))
tangent = tangent / jnp.linalg.norm(tangent, axis=0)
nx, ny = -tangent[1], tangent[0]

# draw the band only where enough members reach to define it
ok = jnp.isfinite(r68)
x, y, r68, nx, ny = track["x"][ok], track["y"][ok], r68[ok], nx[ok], ny[ok]

fig, ax = plt.subplots()
ax.scatter(pos["x"], pos["y"], s=2, c="0.6")
ax.plot(track["x"], track["y"], c="C0")
ax.fill(
    jnp.concatenate([x + r68 * nx, (x - r68 * nx)[::-1]]),
    jnp.concatenate([y + r68 * ny, (y - r68 * ny)[::-1]]),
    color="C0",
    alpha=0.3,
)
ax.set_aspect("equal")
```

The radius is NaN where fewer than the requested fraction of members reach that far,
usually just inside the ends of the track. A member that stops short counts as outside
the tube, so the band never rests on the few members that happen to reach.

## Use the offsets directly

{meth}`~phasecurvefit.TrackEnsemble.offsets` returns each member's offset vector from
the reference, with shape `(n_members, len(gamma), n_spatial_dims)`. Use it for bands
of a different shape, such as a separate width in each direction across a 3-D track.

```{code-cell} python
ensemble.offsets(gamma).shape
```

## With weights

`weights` are resampled along with the samples, so a resampled sample keeps its
weight. A bootstrap treats each sample as one independent draw. The band is then an
honest uncertainty for stars, but not for the pixels of an image, which are not
independent draws ({doc}`../explanation/density-fields`).

## Sample a posterior instead

{func}`~phasecurvefit.sample_track` fits a probabilistic model of the track and
draws from its posterior with NumPyro. It needs the `numpyro` extra
(`pip install "phasecurvefit[numpyro]"`). It starts from the same ordering as
{func}`~phasecurvefit.fit_track` but doesn't train the autoencoder. The SVI steps and
draws here are fewer than the defaults (4000 and 200) to keep the example fast.

```{code-cell} python
post = pcf.sample_track(pos, vel, key=jax.random.key(0), n_steps=2000, n_draws=50)
len(post.members)
```

It returns a {class}`~phasecurvefit.TrackPosterior`. That is a
{class}`~phasecurvefit.TrackEnsemble` whose members are posterior draws and whose
reference is the track at the mean of the drawn parameters, so `radius` and `offsets` work as
above. Each draw also has a width, the spread of the samples about the track.

```{code-cell} python
gamma = jnp.linspace(*post.reference.track_range, 5)
post.radius(gamma, 0.68), post.reference.width(gamma)
```

By default it fits a full-rank Gaussian to the posterior with SVI. `method="nuts"`
runs two NUTS chains instead. That is slower, and it warns when the chains disagree,
which happens when they settle in different modes.

`weights` act as counts here: a sample of weight 2 counts as two samples, so scaling
every weight changes the band. Use counts, such as the number of stars in each pixel,
and not a density normalised to some peak.

With `unxt` Quantities the draws return Quantities. Pass the unit system as
`metadata=pcf.StateMetadata(usys=...)`, as for the orderers.
