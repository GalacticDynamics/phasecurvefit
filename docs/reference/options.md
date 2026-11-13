# Options and defaults

Every option of the orderers and of autoencoder training, with its default. Pass
options as keywords. The {doc}`api` has the full signatures, and the how-to guides
({doc}`../how-to/index`) show how to choose values.

## `LocalFlowOrderer`

{class}`~phasecurvefit.orderers.LocalFlowOrderer`: the local-flow walk
({doc}`../how-to/local-flow-walk`).

| Option | Default | Meaning |
|---|---|---|
| `metric_scale` | `1.0` | the metric's scale: a length for the default metric, a time for `FullPhaseSpaceDistanceMetric`, ignored by `SpatialDistanceMetric` ({doc}`../how-to/tune-metric`) |
| `config` | `WalkConfig()` | the distance metric and neighbour-query strategy, see below |
| `start_idx` | `None` | index of the first sample; `None` takes it from a previous chained stage, or else uses 0 |
| `direction` | `"forward"` | `"forward"` follows the velocities, `"backward"` goes against them, `"both"` walks each way from `start_idx` and joins the two |
| `max_dist` | `inf` | stop when the next step would be longer than this, in position |
| `terminate_indices` | `None` | a `frozenset` of indices at which the walk stops |
| `n_max` | `None` | the most samples the walk visits |
| `nan_policy` | `"raise"` | what to do with NaN inputs |

## `WalkConfig`

{class}`~phasecurvefit.WalkConfig`: the walk's metric and query strategy.

| Option | Default | Meaning |
|---|---|---|
| `metric` | `AlignedMomentumDistanceMetric()` | how candidate points are scored ({doc}`../explanation/walk`) |
| `strategy` | `BruteForce()` | which candidates are scored: all of them, or the `k` nearest with `KDTree(k=...)` ({doc}`../how-to/kdtree`) |

## `MSTOrderer`

{class}`~phasecurvefit.orderers.MSTOrderer`: the minimum-spanning-tree backbone
({doc}`../how-to/choose-orderer`). The default pipeline uses `jump_cap=inf`,
`velocity_factor=30.0`, `orient_by_velocity=True`, `on_disconnected="connect"` and
`nan_policy="omit"` instead of the defaults below.

| Option | Default | Meaning |
|---|---|---|
| `k` | `10` | neighbours each sample links to |
| `jump_cap` | `3.0` | cut edges longer than this, in position (in the unit system's length unit for Quantities) |
| `velocity_weight` | `0.0` | weight of the velocity-alignment penalty added to edge lengths, in position |
| `velocity_factor` | `0.0` | the same penalty in units of the median neighbour distance |
| `sever_cos_threshold` | `None` | drop edges whose velocity cosine is below this |
| `orient_by_velocity` | `False` | flip the ordering so $\gamma$ increases along the mean velocity |
| `on_disconnected` | `"raise"` | for a disconnected graph: `"raise"`, `"warn"`, `"largest"` or `"connect"` |
| `edge_clip_sigma` | `None` | clip edges whose length is an outlier at this many sigma |
| `edge_clip_max_iters` | `5` | most rounds of edge clipping |
| `neighbors` | `None` | the nearest-neighbour backend; `None` picks SciPy for concrete inputs and `BucketKDTree` for traced ones ({doc}`../how-to/jax`) |
| `nan_policy` | `"raise"` | what to do with NaN inputs |

## `RidgeOrderer`

{class}`~phasecurvefit.orderers.RidgeOrderer`: a backbone along the ridge of the
weighted sample density ({doc}`../how-to/choose-orderer`). It does not use
velocities, and it runs on the host, so it can't be traced under `jit`.

| Option | Default | Meaning |
|---|---|---|
| `bandwidth` | `None` | width of the Gaussian kernel for the density, in position, roughly the stream's half-width (in the unit system's length unit for Quantities); `None` estimates it as the RMS distance of the samples from the ridge, iterating from a tenth of the data's extent |
| `k` | `8` | neighbours each sample links to in the graph the backbone runs on |
| `core` | `0.2` | density, as a fraction of its peak, above which a sample can be an end of the backbone |
| `trim` | `0.5` | cut the backbone where the density along it falls below this fraction of its median |

## `SOMOrderer`

{class}`~phasecurvefit.orderers.SOMOrderer`: the self-organizing map
({doc}`../how-to/som`). The default pipeline uses `n_prototypes=15`.

| Option | Default | Meaning |
|---|---|---|
| `n_prototypes` | `25` | lattice points |
| `metric` | `None` | `None` follows the previous stage: phase space after a velocity-aware stage, position otherwise |
| `metric_scale` | `None` | `None` derives it from the data when the metric uses velocity |
| `n_epochs` | `10` | training epochs |
| `sigma_start` | `None` | starting neighbourhood width in lattice units; `None` uses `max(n_prototypes / 4, sigma_end)` on its own and `sigma_end` when chained |
| `sigma_end` | `0.7` | final neighbourhood width in lattice units |
| `densify_factor` | `5` | backbone samples per lattice segment |
| `orient_by_velocity` | `False` | flip the ordering so $\gamma$ increases along the mean velocity |
| `outlier_mixture` | `None` | a `phasecurvefit.som.SOMMixture` to reject outliers, see below |
| `nan_policy` | `"raise"` | `"raise"` or `"omit"` for NaN inputs |

### `SOMMixture`

{class}`~phasecurvefit.som.SOMMixture`: outlier rejection for the SOM, a curve
or background mixture ({doc}`../how-to/som`).

| Option | Default | Meaning |
|---|---|---|
| `background_density` | `None` | density of the flat background, per unit position volume; `None` uses one over the volume of the positions' bounding box |
| `background_inflate` | `1.0` | scale each side of that bounding box by this; raise it when the data do not fill the field |

## `fit_track`

{func}`~phasecurvefit.fit_track`: order, then train an autoencoder, in one call
({doc}`../how-to/autoencoder`).

| Option | Default | Meaning |
|---|---|---|
| `key` | required | random key, split for the model and for training |
| `metadata` | `None` | passed to the default pipeline |
| `weights` | `None` | per-sample weights, passed to the default pipeline and on to training |
| `n_prototypes` | `15` | SOM lattice points; with fewer samples the SOM stage is skipped |
| `outlier_mixture` | `SOMMixture()` | a `SOMMixture` for the SOM stage, to leave out of the ordering samples it calls background; `None` keeps every sample ({doc}`../explanation/autoencoder`) |
| `training_config` | `None` | a `TrainingConfig`; `None` uses its defaults |

## `bootstrap_track`

{func}`~phasecurvefit.bootstrap_track`: a fit and fits to bootstrap resamples, for the
track's uncertainty ({doc}`../how-to/track-uncertainty`). It takes every option of
`fit_track` and passes it to each fit.

| Option | Default | Meaning |
|---|---|---|
| `key` | required | random key, split for the reference fit and for each member |
| `n_samples` | `20` | number of resamples; each costs one `fit_track` |
| `weights` | `None` | per-sample weights, resampled with the samples |

{class}`~phasecurvefit.TrackEnsemble`, which it returns:

| Method | Option | Default | Meaning |
|---|---|---|---|
| `offsets(gamma)` | `n_dense` | `1000` | points per member track searched for the nearest one |
| `radius(gamma, level)` | `level` | `0.68` | fraction of members inside the tube |

## `sample_track`

{func}`~phasecurvefit.sample_track`: a posterior for the track, with NumPyro
({doc}`../how-to/track-uncertainty`). Needs the `numpyro` extra.

| Option | Default | Meaning |
|---|---|---|
| `key` | required | random key, split for the inference and for drawing |
| `method` | `"svi"` | `"svi"`: a full-rank Gaussian fitted to the posterior; `"nuts"`: two NUTS chains, with a warning when they disagree |
| `metadata` | `None` | passed to the default pipeline; Quantity inputs need its unit system |
| `weights` | `None` | per-sample weights, which act as counts |
| `ordering` | `None` | the ordering to start from; `None` runs the default pipeline |
| `n_draws` | `200` | posterior draws returned |
| `n_steps` | `4000` | SVI steps |
| `n_warmup` | `500` | NUTS warmup steps per chain |
| `n_basis` | `16` | basis functions per Gaussian-process channel; a stream that crosses itself may need `32` |
| `n_grid` | `256` | points of the $\gamma$ grid each sample's $\gamma$ is summed over; with `n_basis=32` use `512` |
| `background` | `True` | a flat background over the samples' bounding box, with a latent fraction |
| `line_density` | `True` | a latent density of samples along the track |
| `width_varies` | `True` | a width that changes along the track |
| `velocity_tie` | `0.05` | tie the velocities' mean to the track's tangent, with this much slack; `0` is a hard tie, `None` fits velocities freely |

{class}`~phasecurvefit.TrackPosterior`, which it returns, is a
{class}`~phasecurvefit.TrackEnsemble`, so it has `offsets` and `radius`. Its members
are {class}`~phasecurvefit.TrackDraw`s, each with `width(gamma)`.

## `TrainingConfig`

{class}`~phasecurvefit.nn.TrainingConfig`: autoencoder training
({doc}`../how-to/autoencoder`). Phase 1 trains the encoder, phase 2 the decoder,
phase 3 both ({doc}`../explanation/autoencoder`).

| Option | Default | Meaning |
|---|---|---|
| `optimizer` | the library's default Optax optimizer | Optax optimizer for all phases |
| `batch_size` | `100` | samples per batch |
| `show_pbar` | `True` | show a progress bar |
| `member_threshold` | `0.5` | membership probability above which a sample counts as a member |
| `n_epochs_encoder` | `800` | phase 1 epochs |
| `lambda_prob` | `1.0` | weight of the membership loss terms |
| `n_epochs_decoder` | `100` | phase 2 epochs |
| `n_epochs_both` | `200` | phase 3 epochs |
| `lambda_q` | `1.0` | weight of position reconstruction in phase 3 |
| `lambda_p` | `(1.0, 5.0)` | `(start, stop)` of the velocity-alignment weight, ramped over phase 3 |
| `weight_by_density` | `False` | weight samples by inverse density: `True`, or a mapping of keyword arguments passed to the density estimate; use with care |
| `freeze_encoder_final_training` | `None` | freeze the encoder during phase 3; `None` freezes it when `membership` is set, and then its membership head still trains |
| `membership` | `None` | a `MixtureMembershipConfig` to turn on outlier rejection, see below |
| `track_ends` | `"half_max"` | how `track_range` finds the ends: `"half_max"` where the line density falls to half, or `"mask"` from the caps of a mask made by thickening a curve ({doc}`../explanation/density-fields`) |

## `MixtureMembershipConfig`

{class}`~phasecurvefit.nn.MixtureMembershipConfig`: the stream-plus-background
mixture model ({doc}`../how-to/outliers`).

| Option | Default | Meaning |
|---|---|---|
| `sigma_init` | `0.1` | starting stream half-width, in the normalised coordinates the networks see |
| `sigma_min` | `0.001` | floor on the width |
| `sigma_ceiling` | `(0.5, 0.1)` | `(start, stop)` of the width ceiling, shrunk geometrically over training; the trained width stays capped at `stop` |
| `warmup_frac` | `0.3` | fraction of training spent ramping the membership term in |
| `background_density` | `None` | the background density $\rho_{\mathrm{bg}}$; `None` computes it from the extent of the training positions |
| `background_inflate` | `1.0` | inflation of that extent when `background_density` is `None`; raise it if the stars do not fill the survey footprint |
| `width_size` | `32` | width of the network that predicts the stream width |
| `depth` | `2` | depth of that network |
| `lambda_velocity` | `1.0` | weight of the velocity-alignment term, reweighted by membership; 0 turns it off |
