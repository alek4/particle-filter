# Particle filter

Code: [particle_filter.py](particle_filter.py)

**Goal:** estimate where the target is, (x, y). The robot pose is known.

**Idea:** keep many guesses ("particles") of the target position. Each has a
weight = how much we believe it. Together they approximate the probability
distribution of the target position: where particles are dense and heavy, the
target is likely to be.

State:

```
particles : N x 2 array, each row a guess (x, y)
weights   : N array, sums to 1
```

The filter never sees the true target position. It only sees the segmentator's
output (the detections) and the robot pose.

## Algorithm

**Algorithm 1:** Particle filter for the target position

**Input:** table $T$, number of particles $N$; at each frame $t$ the robot pose
$(\mathbf{p}_t, \theta_t)$ and the detections $Z_t = \{ z_t^1, \dots, z_t^{M_t} \}$,
each $z = (\beta, \rho)$ = (bearing, range)\
**Output:** particles $\{ (\mathbf{s}^i, w^i) \}_{i=1}^{N}$

1: **for** $i = 1, \dots, N$ **do** &emsp; ▷ *prior*\
2: &emsp; $\mathbf{s}^i \sim \mathcal{U}(T)$, &ensp; $w^i \leftarrow 1/N$\
3: **for each** frame $t$ **do**\
4: &emsp; **for** $i = 1, \dots, N$ **do** &emsp; ▷ *resample*\
5: &emsp;&emsp; draw $a$ with $P(a) = w^a$; &ensp; $\tilde{\mathbf{s}}^i \leftarrow \mathbf{s}^a$\
6: &emsp; **for** $i = 1, \dots, N$ **do** &emsp; ▷ *jitter*\
7: &emsp;&emsp; $\mathbf{s}^i \leftarrow \operatorname{clip}_T\!\left( \tilde{\mathbf{s}}^i + \boldsymbol{\varepsilon}^i \right)$, &ensp; $\boldsymbol{\varepsilon}^i \sim \mathcal{N}\!\left(\mathbf{0}, \operatorname{diag}(\mathbf{h})^2\right)$, &ensp; $\mathbf{h} = \max\!\left(N^{-1/6} \operatorname{std}(\tilde{\mathbf{s}}),\ h_{\min}\right)$\
8: &emsp; **for** $i = 1, \dots, N$ **do** &emsp; ▷ *update*\
9: &emsp;&emsp; $\rho^i \leftarrow \lVert \mathbf{s}^i - \mathbf{p}_t \rVert$, &ensp; $\beta^i \leftarrow \operatorname{wrap}\!\left( \operatorname{atan2}(\mathbf{s}^i - \mathbf{p}_t) - \theta_t \right)$\
10: &emsp;&emsp; **if** $\rho^i > R$ **or** $|\beta^i| > \varphi / 2$ **then** &emsp; ▷ *not visible: baseline*\
11: &emsp;&emsp;&emsp; $w^i \leftarrow 1$\
12: &emsp;&emsp; **else** &emsp; ▷ *visible*\
13: &emsp;&emsp;&emsp; $p_{TP} \leftarrow p_{\text{near}} + (p_{\text{far}} - p_{\text{near}})\, \rho^i / R$\
14: &emsp;&emsp;&emsp; $w^i \leftarrow \underbrace{(1 - p_{TP})}_{\text{missed}} + \displaystyle\sum_{j=1}^{M_t} \underbrace{p_{TP}\, \frac{\mathcal{N}(\beta_j - \beta^i;\, 0, \sigma_\beta^2)\, \mathcal{N}(\rho_j - \rho^i;\, 0, \sigma_\rho(\rho^i)^2)}{\lambda_c}}_{z^j \text{ is the target}}$\
15: &emsp; **for** $i = 1, \dots, N$ **do** &emsp; ▷ *normalize*\
16: &emsp;&emsp; $w^i \leftarrow w^i \,/\, \sum_{k=1}^{N} w^k$

where

| Symbol | Meaning | Value |
|---|---|---|
| $R$, $\varphi$ | FOV range and opening angle | 200 px, 60° |
| $p_{\text{near}}$, $p_{\text{far}}$ | detection probability at distance 0 and $R$ | 0.95, 0.5 |
| $\sigma_\beta$ | bearing noise | 3° |
| $\sigma_\rho(\rho)$ | range noise | $\max(0.1\,\rho,\ 1)$ px |
| $\lambda_c$ | false-positive density, $\lambda_{FP} / (\varphi R)$ | $\lambda_{FP} = 0.2$ per frame |
| $\mathbf{h}$, $h_{\min}$ | jitter std per axis (proportional to the cloud spread), its lower bound | $0.35 \cdot$ spread, 0.2 px |

The target estimate, if needed, is the weighted mean
$\hat{\mathbf{s}} = \sum_i w^i \mathbf{s}^i$.

## 1. Prior

We know the target is on a table (a rectangle), but not where on it, so every
position on the table is equally likely:

```
N = 500
for i in 1..N:
    particles[i] = (Uniform(table.left, table.right), Uniform(table.top, table.bottom))
    weights[i]   = 1 / N
```

The table contains the target, but at a random spot (not centred), so the prior
gives no hint about the exact position: the filter must find it.

## 2. Each segmentator frame: `step(robot, measurement)`

```
resample()
jitter()
update(robot, detections)
normalize()
```

No predict step: the target does not move.

### Resample

Pick N particles from the current ones, each chosen with probability = its
weight (good guesses get copied, bad ones disappear). Then all weights = 1/N.

```
idx       = N random indices, P(i) = weights[i]
particles = particles[idx]
weights   = 1/N
```

### Jitter

Resampling makes exact copies; without anything moving them, the number of
distinct positions shrinks every frame until a few points are left. Jitter adds
a little noise so copies spread apart and can explore nearby positions:

Two versions, chosen by `JITTER_MODE` (or press **J** in the simulator):

```
# fixed: jitter_fixed()
particles += Normal(0, JITTER)             # JITTER = 3 px

# adaptive (default): jitter_adaptive()
h          = N^(-1/6) * std(particles)     # per axis; 0.35 * spread for N = 500
h          = max(h, JITTER_MIN)            # never exactly 0
particles += Normal(0, h)

# both
clip particles to the table                # the target cannot be off it
```

The adaptive noise is **proportional to the current spread** of the cloud
("regularized particle filter"): big at the start when particles cover the
table and need to explore, small once they converge. The fixed jitter puts a
floor on how tight the cloud can get: with 3 px the cloud never got below
~7 px wide (entropy stuck at ~0.4).

Remaining limit: every jitter adds ~12% (0.35²) of the cloud's variance, which
partly erases what previous frames established, so the filter effectively
remembers only the last ~8 frames (spread ~5 px at 145 px distance).

### Update: `likelihood(particle, robot, detections)`

For each particle we ask: **if the target were here, how well would that
explain what the segmentator just reported?** Weights are *set* to this
(after resampling they are all 1/N, so setting = multiplying).

Only **relative** weights matter, because we normalize afterwards. So we pick a
baseline and measure everything against it.

**Step 1: where would the target appear to the robot if it were here?**

```
r = distance robot -> particle
b = angle robot -> particle, relative to the robot heading
visible = r <= fov_range and |b| <= fov/2
```

**Case A: particle not visible → weight 1 (the baseline).**
If the target were here the robot could not see it, so every detection in this
frame must be a false positive.

**Case B: particle visible.** Two ways the frame could have happened:

- **B1, the target was missed** (probability `P_FN = 1 - P_TP`). Then all
  detections are false positives, exactly like the baseline. Contribution:
  `P_FN`.
- **B2, the target was detected, and it is detection `z`.** Contribution, for
  each detection `z`:

  ```
  P_TP * closeness(z, particle) / fp_density
  ```

  - `closeness`: how close `z` is to where this particle would appear.
    A Gaussian on the bearing difference times a Gaussian on the range
    difference: big if `z` lands right on the particle, ~0 if it is far.
  - `fp_density`: how likely a false positive is to land at any given spot
    (false positives are spread uniformly over the FOV).
  - The ratio says: *how much more likely is it that `z` was produced by a
    target here, than that `z` was just a random false positive?*

```
if not visible:
    w = 1
else:
    P_TP = P_TP_NEAR + (P_TP_FAR - P_TP_NEAR) * r / fov_range   # drops with distance
    w = 1 - P_TP                                                 # B1: missed
    for each detection z:                                        # B2: z is the target
        closeness = Normal(z.bearing - b; 0, SIGMA_BEARING) * Normal(z.range - r; 0, SIGMA_RANGE)
        w += P_TP * closeness / fp_density

fp_density = FALSE_RATE / (fov * fov_range)
```

**Worked example.** Robot at (500, 200) looking right, one detection straight
ahead at range 100:

| particle                         | weight, with that detection | weight, nothing detected |
|----------------------------------|-----------------------------|--------------------------|
| exactly on the detection (r=100) | **231**                     | 0.28                     |
| 20 px further away (r=120)       | 45                          | 0.32                     |
| 10 px sideways (~6°)             | 38                          | 0.28                     |
| visible, far from it (r=50)      | 0.16                        | 0.16                     |
| not visible                      | 1                           | 1                        |

- With a detection, particles right on it win by a lot; visible particles far
  from it get *less* than the baseline (it looks like the target was missed).
- With nothing detected, visible particles get ~0.3 vs 1: "we looked there and
  saw nothing" pushes particles out of the FOV. That is the old confusion
  matrix row `P_FN` vs `P_TN`.
- Sideways is punished more than further away: bearing is precise (3°),
  range is not (10%).

**Relation to version 1 (below):** with no detections it is the same
confusion matrix. The only addition is that a detection boosts particles
*near it*, instead of every visible particle equally.

### Update, map mode: `likelihood_map(particle, robot, map)`

Used when the segmentator outputs a probability map (see
[SEGMENTATOR.md](SEGMENTATOR.md)). Same structure as above, but instead of
measuring distances to points, each particle reads **the map value `m` at its
own cell**:

```
if not visible:
    w = 1                                                  # baseline, as before
else:
    w = P_FN + P_TP * f_target(m) / f_background(m)

f_target(m)     = Beta(m; 3, 2)                            # values on the target: mostly ~0.6
f_background(m) = (1 - eps) * Beta(m; 1, 15) + eps * f_target(m)
                                                           # mostly ~0.06, but a fraction
                                                           # eps = 0.002 of cells are false blobs
```

- `f_target(m) / f_background(m)`: *how much more typical is this value of a
  target cell than of a background cell?* In practice the two distributions are
  the **histograms of the segmentator's scores** on target pixels and on
  background pixels, measured on labelled data. It is the per-score version of
  the confusion matrix.
- On a background-looking cell the ratio is ~0, so `w = P_FN`: exactly the
  "looked there, saw nothing" case of points mode.
- `eps` plays the role of `fp_density` in points mode: without it a strong value
  would be "impossible" on background, the ratio would explode (10¹² at
  m = 0.9) and a single false blob would grab all the particles in one frame.
  With it the ratio is capped at ~1/eps.

Weights at r = 100 (`P_TP = 0.725`):

| m      | 0.06 | 0.15 | 0.25 | 0.4 | 0.6 | 0.9 |
|--------|------|------|------|-----|-----|-----|
| weight | 0.28 | 0.38 | 1.8  | 60  | 359 | 363 |

Only **one cell per particle** is read, so neighbouring cells of the same blob
are never multiplied together (that would make the filter overconfident).

Result (robot following the estimate, N = 100, 8 runs, last 10 s):

| false positives / frame | points mode error | map mode error |
|---|---|---|
| 0.2 | 5.6 px | 3.8 px |
| 1.0 | 5.6 px | 4.1 px |

Map mode is more precise because it also uses the blob strength and does not
throw away weak evidence.

**Why not just treat `m` as the probability that the target is there?** The
first try did: `w = odds(m) / odds(0.2)`. It failed (error ~100 px, particles
fled the FOV even with the target straight ahead): plain background (0.06)
counted as fairly strong evidence *against* (x0.26 every frame), while a
target blob only gave x4–16. Our map values are not calibrated probabilities,
and neither are a real network's without calibration.

### Normalize

```
weights = weights / sum(weights)
```

## 3. Uncertainty and estimate: `compute_stats()`

Run after every step. The table is split into a grid of `CELL x CELL` cells
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

**Entropy** (yellow plot): how spread out the belief is.

| entropy | meaning |
|---|---|
| 0 | all the weight in one cell: certain |
| ~0.3 | a few neighbouring cells: converged (what a good run reaches) |
| ~0.95 | the uniform prior (not exactly 1: 500 random particles do not fill 240 cells evenly) |
| 1 | perfectly uniform over the table |

Unlike a covariance or a radius, it tells **one wide blob** (high) apart from
**two tight clusters** (moderate, about one extra bit), which matters because a
false positive can create a temporary second cluster.

**Best cell probability** (blue plot): how sure we are about the reported
cell. It does not get close to 1 even when converged, because the particles are
spread over a few cells (see the jitter section for why the cloud keeps a few px of spread).

**Estimate = best cell**, not the weighted mean of all particles: with two
clusters the overall mean would fall between them, where there is no target.
On screen: yellow square = best cell, yellow cross = estimate.

The panel also shows the error between the estimate and the true target. The
filter does not use it: it is there to check that low entropy really comes
with low error (if entropy is low but the error is large, the filter is
overconfident).

## Known weaknesses

1. **Resampling every frame** loses diversity even when the frame said nothing
   new (e.g. no particle in the FOV).
2. **Assumes errors are independent between frames.** The simulator has
   failure bursts (several misses in a row) the filter does not know about:
   that is a deliberate mismatch to test robustness.

---

## Version 1: confusion matrix only (replaced)

Used only "something detected or not", not where:

| | something detected | nothing detected |
|---|---|---|
| **particle in FOV**     | `P_TP = 0.8`  | `P_FN = 0.2` |
| **particle not in FOV** | `P_FP = 0.18` | `P_TN = 0.82` |

Why it was replaced, tested with a robot standing still, target straight ahead:

| target distance | v1 after 10 s: error / spread | v2 after 1 s: error / spread |
|-----------------|-------------------------------|------------------------------|
| 100 px          | 22 / 31 px                    | 5 / 6 px                     |
| 150 px          | 23 / 34 px                    | 5 / 7 px                     |
| 190 px          | particles left the FOV        | 4 / 7 px                     |

- A detection boosted every visible particle equally, so particles just filled
  the part of the table inside the FOV and never shrank to a point.
- Constant `P_TP = 0.8` was too optimistic far away (real ~0.45), so misses
  outweighed detections and pushed particles out of the FOV even when the
  target was in it.
