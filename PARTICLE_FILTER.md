# Particle filter

Code: [particle_filter.py](particle_filter.py)

**Goal:** estimate where the target is, (x, y). The robot pose is known.

**Idea:** keep many guesses ("particles") of the target position. Each has a
weight = how much we believe it. Together they approximate the probability
distribution of the target position: where particles are dense and heavy, the
target is likely to be.

```
particles : N x 2 array, each row a guess (x, y)
weights   : N array, sums to 1
```

The filter never sees the true target position. It only sees the segmentator's
output (a probability map, see [SEGMENTATOR.md](SEGMENTATOR.md)) and the robot pose.

## Algorithm

**Algorithm 1:** Particle filter for the target position

**Input:** table $T$, number of particles $N$; at each frame $t$ the robot pose
$(\mathbf{p}_t, \theta_t)$ and the segmentation map $M_t$ (one value in $[0, 1]$
per (bearing, range) cell of the FOV)\
**Output:** particles $\{ (\mathbf{s}^i, w^i) \}_{i=1}^{N}$

1: **for** $i = 1, \dots, N$ **do** &emsp; ▷ *prior*\
2: &emsp; $\mathbf{s}^i \sim \mathcal{U}(T)$, &ensp; $w^i \leftarrow 1/N$\
3: **for each** frame $t$ **do**\
4: &emsp; **for** $i = 1, \dots, N$ **do** &emsp; ▷ *jitter (the "motion model" of a static target)*\
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
15: &emsp; compute the estimate from $\{ (\mathbf{s}^i, w^i) \}$ &emsp; ▷ *section 3*\
16: &emsp; **for** $i = 1, \dots, N$ **do** &emsp; ▷ *resample: the new set is unweighted*\
17: &emsp;&emsp; draw $a$ with $P(a) = w^a$; &ensp; $\mathbf{s}^i_{\text{new}} \leftarrow \mathbf{s}^a$

where

| Symbol | Meaning | Value |
|---|---|---|
| $R$, $\varphi$ | FOV range and opening angle | 200 px, 60° |
| $p_{\text{near}}$, $p_{\text{far}}$ | probability the target shows up in the map, at distance 0 and $R$ | 0.95, 0.5 |
| $f_T(m)$ | how typical value $m$ is on the target's cell | $\text{Beta}(m; 3, 2)$ |
| $f_B(m)$ | how typical value $m$ is on a background cell | $(1-\epsilon)\,\text{Beta}(m; 1, 15) + \epsilon\, f_T(m)$, $\epsilon = 0.002$ |
| $\mathbf{h}$, $h_{\min}$ | jitter std per axis (proportional to the cloud spread), its lower bound | $N^{-1/6} \cdot$ spread ($0.46 \cdot$ spread for $N = 100$), 0.2 px |

This is the order of *Probabilistic Robotics* (Thrun, Burgard, Fox), Table 4.3:
sample, weight, resample. The estimate is computed on the weighted set, just
before resampling (section 3).

## 1. Prior

We know the target is on a table (a rectangle), but not where on it, so every
position on the table is equally likely:

```
for i in 1..N:
    particles[i] = (Uniform(table.left, table.right), Uniform(table.top, table.bottom))
    weights[i]   = 1 / N
```

The table contains the target, but at a random spot (not centred), so the prior
gives no hint about the exact position: the filter must find it.

## 2. Each segmentator frame: `step(robot, map)`

```
jitter()
update(robot, map)
normalize()
compute_stats()
resample()
```

The target does not move, so there is no real motion model: the jitter takes
its place.

### Jitter

Resampling makes exact copies; without anything moving them, the number of
distinct positions shrinks every frame until a few points are left. Jitter adds
a little noise so copies spread apart and can explore nearby positions.

Two versions, chosen by `JITTER_MODE` (or press **J** in the simulator):

```
# fixed: jitter_fixed()
particles += Normal(0, JITTER)             # JITTER = 3 px

# adaptive (default): jitter_adaptive()
h          = N^(-1/6) * std(particles)     # per axis; 0.46 * spread for N = 100
h          = max(h, JITTER_MIN)            # never exactly 0
particles += Normal(0, h)

# both
clip particles to the table                # the target cannot be off it
```

The adaptive noise is **proportional to the current spread** of the cloud
("regularized particle filter"): big at the start when particles cover the
table and need to explore, small once they converge. The fixed jitter puts a
floor on how tight the cloud can get. Measured with N = 100, target 145 px away:

| jitter | cloud spread (std) | entropy |
|---|---|---|
| fixed 3 px | ~18 px | ~0.5 |
| adaptive | ~4 px | ~0.17 |

### Update: `likelihood(particle, robot, map)`

For each particle: **how much more likely is this map if the target is at the
particle than if it is not?** Weights are *set* to this, not multiplied: the
particles come from the previous resample, so the old weights are already
encoded in how many copies of each particle there are.

```
r, b = range and bearing of the particle from the robot
if the particle is outside the FOV:
    w = 1                                   # the map says nothing about it
else:
    m           = map value at the particle's cell, clipped to [0.01, 0.99]   # avoids weights of 0 or infinity
    P_TP        = P_TP_NEAR + (P_TP_FAR - P_TP_NEAR) * r / fov_range
    target_here = P_TP * f_target(m) + (1 - P_TP) * f_background(m)   # seen, or missed
    no_target   = f_background(m)
    w           = target_here / no_target
```

- `f_target(m)`, `f_background(m)`: how typical the value `m` is on the
  target's cell and on a background cell. Both are **Beta distributions**
  (distributions over numbers between 0 and 1): background Beta(1, 15) piles up
  near 0 (mean 0.06), target Beta(3, 2) is a hump around 0.6. With a real
  segmentator they would be the **histograms of its scores** on target and
  background pixels, measured on labelled data.
- **Target missed** (`1 - P_TP`): then the cell is just background, so that
  part of `target_here` equals `no_target`.
- **False positives**: `f_background` gives a small weight `eps = 0.002` to
  target-looking values (some background cells are false blobs). Without it a
  strong value would be "impossible" on background, the ratio would explode
  (10¹² at m = 0.9) and one false blob would grab all the particles in a single
  frame. With it the weight is capped at ~1/eps.
- Only **one cell per particle** is read, so neighbouring cells of the
  same blob are never multiplied together (that would make the filter
  overconfident).

Weights for a particle at r = 100 (`P_TP = 0.725`):

| m      | 0.06 | 0.15 | 0.25 | 0.4 | 0.6 | 0.9 |
|--------|------|------|------|-----|-----|-----|
| weight | 0.28 | 0.38 | 1.8  | 60  | 359 | 363 |

- Background-looking cell → 0.28 = `1 - P_TP`: "we looked there and saw
  nothing" pushes particles out of the FOV (outside it they keep 1).
- Blob-looking cell → a few hundred.

Result (robot following the estimate, N = 100, 8 runs, last 10 s): error
3.8 px with 0.2 false-positive blobs per frame, 4.1 px with 1.0.

### Normalize

```
weights = weights / sum(weights)
```

### Resample

Pick N particles from the current ones, each chosen with probability = its
weight (good guesses get copied, bad ones disappear). The weights are not
reset: the new set is unweighted, and the next `update` overwrites them.

```
idx       = N random indices, P(i) = weights[i]
particles = particles[idx]
```

## 3. Uncertainty and estimate: `compute_stats()`

Run in every step, after normalizing and before resampling. The table is split into a grid of `CELL x CELL` cells
(10 px, so 20 x 12 = 240 cells), and each cell gets the total weight of the
particles inside it = the probability that the target is in that cell.

```
for each cell c:
    cell_mass[c] = sum of weights of particles in c

entropy   = - sum over cells with mass > 0 of  cell_mass[c] * log(cell_mass[c])
entropy   = entropy / log(number of cells)          # scale to [0, 1]

best_cell = cell with the largest cell_mass
best_prob = cell_mass[best_cell]
estimate  = weighted mean of the particles inside best_cell
```

**Entropy** (amber plot): how spread out the belief is.

| entropy | meaning |
|---|---|
| 0 | all the weight in one cell: certain |
| ~0.1–0.3 | a few neighbouring cells: converged |
| ~0.8 | the uniform prior with N = 100 (not 1: 100 random particles cannot fill 240 cells evenly) |
| 1 | perfectly uniform over the table |

Unlike a covariance or a radius, it tells **one wide blob** (high) apart from
**two tight clusters** (moderate, about one extra bit), which matters because a
false positive can create a temporary second cluster.

**Best cell probability** (cyan plot): how sure we are about the reported
cell. It rarely reaches 1, because the particles spread over a few cells.

**Estimate = best cell**, not the weighted mean of all particles: with two
clusters the overall mean would fall between them, where there is no target.
On screen: cyan square = best cell, cyan crosshair = estimate.

The panel also shows the error between the estimate and the true target. The
filter does not use it: it is there to check that low entropy really comes
with low error (if entropy is low but the error is large, the filter is
overconfident).

## Known weaknesses

1. **Resampling every frame** loses diversity even when the frame said nothing
   new (e.g. no particle in the FOV), and with jitter it limits how much the
   filter remembers from older frames.
2. **Assumes errors are independent between frames.** The simulator has
   failure bursts (several misses in a row) the filter does not know about:
   a deliberate mismatch to test robustness.
3. **Assumes map cells are independent.** Fine here because each particle reads
   one cell; real segmentation noise comes in patches, which would look like
   extra false blobs.
