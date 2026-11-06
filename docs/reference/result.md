# The ordering result

Every orderer returns an {class}`~phasecurvefit.orderers.OrderingResult`. The
local-flow walk returns `WalkLocalFlowResult`, a subclass with the same fields.

## Fields

| Field | Contents |
|---|---|
| `positions` | the input positions, in input order (not reordered) |
| `velocities` | the input velocities, in input order |
| `indices` | sample indices in visit order, padded with `-1` for unvisited samples to a fixed length; never index with it directly |
| `gamma_range` | the range of $\gamma$ that calling the result accepts; `(-1.0, 1.0)` for the MST, the SOM and the default pipeline |
| `backbone` | an ordered polyline that calling the result interpolates along, or `None` for the walk, which interpolates between the visited samples |
| `backbone_size` | the number of valid leading vertices when `backbone` is padded to a fixed shape, as the MST does under `jit`; `None` when every vertex is valid |
| `chord` | arc length along the curve for each sample, in input order, with `nan` for unvisited samples; all built-in orderers set it, and `None` means an orderer did not |
| `weights` | the per-sample weights passed to `order()`, in input order, or `None` when none were passed; {func}`~phasecurvefit.nn.train_autoencoder` reads them from here |
| `velocity_aware` | whether the orderer's metric uses velocity |

In `indices`, the position in the array is the step along the ordering and the value
is the original sample index:

```text
indices = [3, 7, 1, 5, -1, -1]
#          ^ first is sample 3
#             ^ second is sample 7, and so on
#                      ^ two samples were not visited
```

## Accessors

| Accessor | Returns |
|---|---|
| `ordering` | the visited indices in order: `indices` without the `-1` padding |
| `visited` | a boolean mask over the samples, in input order |
| `n_visited`, `n_skipped` | how many samples were and were not visited |
| `all_visited` | whether every sample was visited |
| `skipped_indices` | the indices of the samples not visited |
| `ordered` | positions and velocities reordered along the ordering |
| `result(gamma)` | positions interpolated at $\gamma$ in `gamma_range` ({doc}`../how-to/interpolate`) |

{func}`~phasecurvefit.order_w` also returns the positions and velocities in order.

## The autoencoder's result

{func}`~phasecurvefit.nn.train_autoencoder` and {func}`~phasecurvefit.fit_track`
return an {class}`~phasecurvefit.nn.AutoencoderResult`. It has `positions`,
`velocities` and `gamma_range` as above, plus:

| Field | Contents |
|---|---|
| `indices` | the indices of the samples whose membership probability is at least `member_threshold`, sorted by $\gamma$; shorter than the number of samples when any are rejected, and never padded with `-1` |
| `gamma` | $\gamma$ for every sample |
| `membership_prob` | the encoder's membership probability for every sample |
| `model` | the trained network |
| `track_range` | the range of $\gamma$ the track covers, for drawing it. Each end is where the weighted number of members per unit length along the track falls below half its value just inside, so the faint tail of a blurred stream doesn't carry the track past the end, while a stream that narrows or fades keeps its ends. Inside `gamma_range`, which still accepts every sample's $\gamma$ |

Calling it, `result(gamma)`, evaluates the fitted track ({doc}`../how-to/autoencoder`).
