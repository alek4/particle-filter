# Particle filter

Code: [`particle_filter.py`](../particle_filter.py)

**Goal:** estimate the target position (x, y) on the table. The robot pose is
known.

The filter never sees the true target position. Its only inputs are the
robot pose and the segmentator's probability map (see
[segmentator.md](segmentator.md)).

## Overview

The belief about the target position is represented by **N particles**, each
one a guess (x, y). Where particles are dense, the target is likely to be.

At the start, the particles are spread **uniformly over the table**: all we
know is that the target is somewhere on it.

Each segmentator frame then runs these steps:

1. **Jitter.** Every particle moves by a small Gaussian noise. The target is
   static, so this takes the place of a motion model. It also spreads apart
   the identical copies created by resampling.
2. **Update.** Every particle gets a weight. Outside the FOV the weight is 1,
   because the map says nothing about that particle. Inside the FOV, the map
   value at the particle's cell is read. The weight is the ratio between how
   likely that value is if the target is there (either seen, or missed with a
   probability that grows with distance) and how likely it is if that cell is
   just background.
3. **Normalize** the weights. Then compute the **uncertainty** (entropy) and
   the **estimate**: the weighted mean of the particles in the most probable
   cell of the table.
4. **Resample.** Draw N new particles from the current ones, each with
   probability equal to its weight. Good guesses get copied, bad ones disappear.

This is the loop of *Probabilistic Robotics* (Thrun, Burgard, Fox), Table 4.3:
sample, weight, resample.

## Algorithm

**Input:** table $T$, number of particles $N$. At each frame $t$: the robot
pose $(\mathbf{p}_t, \theta_t)$ and the segmentation map $M_t$ (one value in
$[0, 1]$ per (bearing, range) cell of the FOV).\
**Output:** weighted particles $\{ (\mathbf{s}^i, w^i) \}_{i=1}^{N}$ and an estimate.

1: **for** $i = 1, \dots, N$ **do** &emsp; ▷ *prior*\
2: &emsp; $\mathbf{s}^i \sim \mathcal{U}(T)$\
3: **for each** frame $t$ **do**\
4: &emsp; **for** $i = 1, \dots, N$ **do** &emsp; ▷ *jitter (stands in for the motion model)*\
5: &emsp;&emsp; $\mathbf{s}^i \leftarrow \operatorname{clip}_T\!\left( \mathbf{s}^i + \boldsymbol{\varepsilon}^i \right)$, &ensp; $\boldsymbol{\varepsilon}^i \sim \mathcal{N}\!\left(\mathbf{0}, \operatorname{diag}(\mathbf{h})^2\right)$, &ensp; $\mathbf{h} = \max\!\left(N^{-1/6} \operatorname{std}(\mathbf{s}),\ h_{\min}\right)$\
6: &emsp; **for** $i = 1, \dots, N$ **do** &emsp; ▷ *update: weights are set, not multiplied*\
7: &emsp;&emsp; $\rho^i \leftarrow \lVert \mathbf{s}^i - \mathbf{p}_t \rVert$, &ensp; $\beta^i \leftarrow \operatorname{wrap}\!\left( \operatorname{atan2}(\mathbf{s}^i - \mathbf{p}_t) - \theta_t \right)$\
8: &emsp;&emsp; **if** $\rho^i > R$ **or** $|\beta^i| > \varphi / 2$ **then** &emsp; ▷ *not visible*\
9: &emsp;&emsp;&emsp; $w^i \leftarrow 1$\
10: &emsp;&emsp; **else** &emsp; ▷ *visible: read the map at the particle's cell*\
11: &emsp;&emsp;&emsp; $m \leftarrow \operatorname{clip}\!\left(M_t(\beta^i, \rho^i),\ 0.01,\ 0.99\right)$, &ensp; $p_{TP} \leftarrow p_{\text{near}} + (p_{\text{far}} - p_{\text{near}})\, \rho^i / R$\
12: &emsp;&emsp;&emsp; $w^i \leftarrow \dfrac{p_{TP}\, f_T(m) + (1 - p_{TP})\, f_B(m)}{f_B(m)}$ &emsp; ▷ *P(m | target here) / P(m | no target here)*\
13: &emsp; **for** $i = 1, \dots, N$ **do** &emsp; ▷ *normalize*\
14: &emsp;&emsp; $w^i \leftarrow w^i \,/\, \sum_{k=1}^{N} w^k$\
15: &emsp; compute entropy and estimate from $\{ (\mathbf{s}^i, w^i) \}$ &emsp; ▷ *see [Uncertainty and estimate](#uncertainty-and-estimate)*\
16: &emsp; **for** $i = 1, \dots, N$ **do** &emsp; ▷ *resample: the new set is unweighted*\
17: &emsp;&emsp; draw $a$ with $P(a) = w^a$; &ensp; $\mathbf{s}^i_{\text{new}} \leftarrow \mathbf{s}^a$

| Symbol | Meaning | Value |
|---|---|---|
| $R$, $\varphi$ | FOV range and opening angle | 200 px, 60° |
| $p_{\text{near}}$, $p_{\text{far}}$ | probability the target shows up in the map, at distance 0 and $R$ | 0.95, 0.5 |
| $f_T(m)$ | how typical value $m$ is on the target's cell | $\text{Beta}(m; 3, 2)$ |
| $f_B(m)$ | how typical value $m$ is on a background cell | $(1-\epsilon)\,\text{Beta}(m; 1, 15) + \epsilon\, f_T(m)$, $\epsilon = 0.002$ |
| $\mathbf{h}$, $h_{\min}$ | jitter std per axis (proportional to the cloud spread), lower bound | $N^{-1/6} \cdot$ spread, 0.2 px |

## Steps in detail

```
step(robot, map):
    jitter()
    update(robot, map)
    normalize()
    compute_stats()
    resample()
```

### Prior

The target is known to be on the table (a rectangle), but not where. Every
position on it is equally likely:

```
particles[i] = (Uniform(table.left, table.right), Uniform(table.top, table.bottom))
```

The table is placed at a random offset around the target, so the prior gives
no hint about the exact position.

### Jitter

Resampling makes exact copies. If nothing moved them, the number of distinct
positions would shrink every frame until only a few points were left. Jitter
adds a little noise, so copies spread apart and explore nearby positions.

Two versions, chosen by `JITTER_MODE` (press **J** in the simulator to switch):

```
# fixed
particles += Normal(0, JITTER)             # JITTER = 3 px

# adaptive (default)
h          = N^(-1/6) * std(particles)     # per axis; 0.46 * spread for N = 100
h          = max(h, JITTER_MIN)            # never exactly 0
particles += Normal(0, h)

# both
clip particles to the table                # the target cannot be off it
```

The adaptive noise is **proportional to the current spread** of the cloud, as
in a regularized particle filter. It is large at the start, when particles
cover the table and need to explore, and small once they converge. Fixed jitter
puts a floor on how tight the cloud can get. Measured with N = 100 and the
target 145 px away:

| jitter | cloud spread (std) | entropy |
|---|---|---|
| fixed 3 px | ~18 px | ~0.5 |
| adaptive | ~4 px | ~0.17 |

### Update (likelihood)

For each particle the question is: **how much more likely is this map if the
target is at the particle than if it is not?**

```
r, b = range and bearing of the particle from the robot
if the particle is outside the FOV:
    w = 1                                   # the map says nothing about it
else:
    m           = map value at the particle's cell, clipped to [0.01, 0.99]
    P_TP        = P_TP_NEAR + (P_TP_FAR - P_TP_NEAR) * r / fov_range
    target_here = P_TP * f_target(m) + (1 - P_TP) * f_background(m)   # seen, or missed
    no_target   = f_background(m)
    w           = target_here / no_target
```

- **Weights are set, not multiplied.** The particles come from the previous
  resample, so the old weights are already encoded in how many copies of each
  particle exist. Multiplying by them again would count that evidence twice.
- **`f_target`, `f_background`** say how typical the value `m` is on the
  target's cell and on a background cell. Both are Beta distributions
  (distributions over [0, 1]). Background Beta(1, 15) piles up near 0 (mean
  0.06). Target Beta(3, 2) is a hump around 0.6. With a real segmentator they
  would be the histograms of its scores on target and background pixels,
  measured on labelled data.
- **Target missed** (`1 - P_TP`): the cell then looks like background, so that
  part of `target_here` equals `no_target`.
- **False positives:** `f_background` puts a small weight `ε = 0.002` on
  target-looking values, because some background cells are false blobs.
  Without it, a strong value would be "impossible" on background, the ratio
  would explode (10¹² at m = 0.9) and a single false blob would capture all the
  particles in one frame. With it, the weight is capped at about 1/ε.
- **One cell per particle.** Each particle reads one cell only, so neighbouring
  cells of the same blob are never multiplied together, which would make the
  filter overconfident.
- **Clipping** `m` to [0.01, 0.99] keeps the Beta densities finite, so weights
  are never 0 or infinite.

Weights for a particle at r = 100 (`P_TP = 0.725`):

| m      | 0.06 | 0.15 | 0.25 | 0.4 | 0.6 | 0.9 |
|--------|------|------|------|-----|-----|-----|
| weight | 0.28 | 0.38 | 1.8  | 60  | 359 | 363 |

- A background-looking cell gives 0.28 = `1 - P_TP`. "We looked there and saw
  nothing" pushes particles out of the FOV, where they keep weight 1.
- A blob-looking cell gives a few hundred.

Result with the robot following the estimate (N = 100, 8 runs, last 10 s):
mean error 3.8 px with 0.2 false-positive blobs per frame, 4.1 px with 1.0.

### Normalize

```
weights = weights / sum(weights)
```

### Resample

Draw N particles from the current ones, each with probability equal to its
weight:

```
idx       = N random indices, P(i) = weights[i]
particles = particles[idx]
```

The weights are **not** reset. The new set is unweighted (the weights now live
in the number of copies), and the next `update` overwrites them.

## Uncertainty and estimate

`compute_stats()` runs every frame, after normalizing and before resampling.
The table is split into a grid of `CELL × CELL` cells (10 px, so 20 × 12 = 240
cells). Each cell gets the total weight of the particles inside it, which is
the probability that the target is in that cell.

```
cell_mass[c] = sum of the weights of the particles in c

entropy   = - sum over cells with mass > 0 of  cell_mass[c] * log(cell_mass[c])
entropy   = entropy / log(number of cells)          # scale to [0, 1]

best_cell = cell with the largest cell_mass
best_prob = cell_mass[best_cell]
estimate  = weighted mean of the particles inside best_cell
```

**Entropy** (amber plot) measures how spread out the belief is:

| entropy | meaning |
|---|---|
| 0 | all the weight in one cell: certain |
| ~0.1–0.3 | a few neighbouring cells: converged |
| ~0.8 | the uniform prior with N = 100 (not 1, because 100 random particles cannot fill 240 cells evenly) |
| 1 | perfectly uniform over the table |

Unlike a covariance or a radius, entropy tells **one wide blob** (high) apart
from **two tight clusters** (moderate, about one extra bit). This matters
because a false positive can create a temporary second cluster.

**Best cell probability** (cyan plot) says how sure the filter is about the
reported cell. It rarely reaches 1, because the particles spread over a few
cells.

**The estimate is the best cell**, not the weighted mean of all particles. With
two clusters, the overall mean would fall between them, where there is no
target. On screen: cyan square = best cell, cyan crosshair = estimate.

The panel also shows the error between the estimate and the true target. The
filter does not use it. It is there to check that low entropy really comes
with low error: if entropy is low but the error is large, the filter is
overconfident.

## Known weaknesses

1. **Resampling every frame** loses diversity even when the frame brought no
   new information (for example, no particle in the FOV). Together with
   jitter, it limits how much the filter remembers from older frames.
2. **Errors are assumed independent between frames.** The simulator produces
   failure bursts (several misses in a row) that the filter does not model.
   This mismatch is deliberate, to test robustness.
3. **Map cells are assumed independent.** This is fine here because each
   particle reads a single cell. Real segmentation noise comes in patches,
   which would look like extra false blobs.
