# The MST backbone

{class}`~phasecurvefit.orderers.MSTOrderer` orders samples along the backbone of a
minimum spanning tree. It is the first stage of the default pipeline, and it needs no
start point. {doc}`../how-to/choose-orderer` shows how to set it up.

## From a graph to an ordering

1. Each sample is joined to its `k` nearest neighbours in position, and edges longer
   than `jump_cap` are cut.
2. The minimum spanning tree keeps the shortest set of edges that connects every
   sample without loops.
3. The longest path through the tree, its diameter, runs from one end of the curve to
   the other. This backbone is what finds the ends without a start point.
4. Each sample is placed by its arc length along the backbone, which becomes the
   ordering coordinate $\gamma \in [-1, 1]$.

The backbone is kept on the result as `result.backbone`, and calling the result
interpolates along it. Each sample's arc length, measured at its nearest backbone
vertex, is kept as `result.chord`. A sample off to the side of the curve, such as an
interloper the tree threads in, gets the arc length of the backbone next to it, so it
adds no length of its own.

Ordering from end to end with short steps is what a near-closed loop needs. A walk
on such a loop has to start at the right place, and where the velocity reverses at a
progenitor it must start there and walk both ways. The MST finds both ends of the
loop itself.

## Velocity, when the curve crosses itself

By default the MST uses positions only. Where two parts of a curve pass close
together, the tree can then join them, and the backbone short-cuts from one part to
the other. Three options bring the velocities in, each using the cosine
$\cos(v_i, v_j)$ between the two samples' velocities on an edge:

- `velocity_weight` adds a penalty to the edge length that grows as the velocities
  turn apart, so close arms moving in opposite directions are not joined.
  `velocity_factor` is the same penalty measured in median neighbour distances, so
  it needs no units. The penalty is added, not multiplied: at a crossing the edges
  between the branches can be far shorter than a typical edge, and multiplying a
  short length still leaves it short.
- `sever_cos_threshold` removes edges whose velocities point too far apart, cutting
  the seam where a near-closed loop's velocity reverses and the edges between
  branches at a crossing.
- `orient_by_velocity` flips the ordering so $\gamma$ increases along the mean
  velocity, giving it a physically meaningful direction.

On a curve that crosses itself many times these options are necessary, but a walk
that follows the velocity is usually the better tool there
({doc}`walk`, {doc}`../tutorials/epitrochoid_mst`). The MST suits single,
near-closed loops best.

## Why the default pipeline configures it differently

`MSTOrderer()` is a tool to tune: its default `jump_cap=3.0` is an absolute length,
and by default a disconnected graph raises. The default pipeline instead has to work
on any data without tuning. It sets `jump_cap=inf`, so no edge is cut whatever the
data's scale; `velocity_factor=30`, so where a stream crosses itself the tree
crosses straight over instead of turning onto the other branch;
`orient_by_velocity=True`, so $\gamma$ follows the flow; and
`on_disconnected="connect"`, so a gap in a stream that splits the neighbour graph is
bridged along the shortest links instead of losing one side. Because the MST used
velocities, the SOM after it does too ({doc}`som`). A sample with a missing velocity
joins the tree as a leaf and is skipped when orienting.

The value 30 was chosen on a looped test track that crosses itself once, drawn 8
times with different noise. With positions alone the default turned onto the other
branch at the crossing every time: the rank correlation between the ordering and
the true position along the track was 0.60 to 0.91. With `velocity_factor=30` it
was 0.997 to 1.000, and values from 10 to 100 all crossed correctly. On tracks that
do not cross themselves (an arc, a helix, an arc with a gap, and the mock stream in
the tutorials) the ordering did not change.

## Rejecting outliers

An MST threads every sample into the tree, including interlopers, along their
cheapest edge. `edge_clip_sigma` clips edges whose length is an outlier among the
tree's edges, iterating up to `edge_clip_max_iters` times, and leaves the samples it
cuts off unvisited. A {class}`~phasecurvefit.orderers.SOMOrderer` chained after it
works only on the samples the MST kept ({doc}`som`).

## Running under JAX

The graph stage (the tree, its components, the diameter path, edge clipping and
bridging) runs in pure JAX, so the MST works under `jit`, `vmap` and `grad` without
calling back to the host. The neighbour search uses SciPy on concrete inputs, which
is fastest on CPU but cannot be traced, and the JAX-native
{class}`~phasecurvefit.neighbors.BucketKDTree` on traced inputs
({doc}`../how-to/jax`).
