# Segmentator model

Code: [segmentator.py](segmentator.py)

A real segmentator outputs an image with a probability per pixel. We simulate
that image **already projected on the ground**: a polar grid over the FOV,
60 bearing cells (1° each) x 50 range cells (4 px each), one value in [0, 1]
per cell = how much that cell looks like the target.

- **bearing**: angle relative to the robot heading (rad).
- **range**: distance from the robot (px).

This file is the **simulator only**: it is the only code that knows the true
target position. What the *filter* believes about the segmentator is in
[PARTICLE_FILTER.md](PARTICLE_FILTER.md). The filter does not know about
failure bursts or the exact blob shapes, on purpose: if it used exactly the
model that generated the data, the test would be too optimistic.

## Parameters

| Name               | Default | Meaning                                                     |
|--------------------|---------|-------------------------------------------------------------|
| `p_near`           | 0.95    | probability the target shows up when very close            |
| `p_far`            | 0.5     | same, at the edge of the FOV range                          |
| `sigma_bearing`    | 3°      | noise on the blob centre, bearing                           |
| `sigma_range_frac` | 0.10    | noise on the blob centre, range = 10% of the distance       |
| `false_rate`       | 0.2     | average false-positive blobs per frame                      |
| `p_fail`           | 0.02    | per frame, chance a failure burst starts                    |
| `p_recover`        | 0.2     | per frame, chance a burst ends (mean length 5 frames)       |
| `OBJECT_RADIUS`    | 8 px    | size of the target, sets the size of its blob               |
| `BACKGROUND_BETA`  | (1, 15) | background cell values ~ Beta(1, 15): mostly ~0.06          |

## `sense(robot, target)`

```
# 0. failure bursts (two-state switch)
if failing:  failing stays true unless random() < p_recover
else:        failing becomes true if random() < p_fail

# 1. background: every cell gets a small random value
values[every cell] = Beta(1, 15)

# 2. the real target
r, b = range and bearing of the target from the robot
P_d  = p_near + (p_far - p_near) * r / fov_range           # harder to see when far
if target in FOV and not failing and random() < P_d:
    b_seen = b + Normal(0, sigma_bearing)
    r_seen = max(0, r + Normal(0, max(sigma_range_frac * r, 1)))   # noise at least 1 px
    peak   = Uniform(0.6, 1.0) * (1 - 0.4 * r_seen / fov_range)    # weaker when far
    add_blob((b_seen, r_seen), peak)

# 3. false positives: blobs at random places
repeat Poisson(false_rate) times:
    add_blob(random place in the FOV, peak = Uniform(0.4, 0.9))

return values

add_blob(centre, peak):
    every cell = max(cell, peak * Gaussian bump around centre)
    bump width: OBJECT_RADIUS in range, OBJECT_RADIUS / distance in bearing
                (the same object covers fewer pixels when far), at least one cell
```

On screen the map is a green heatmap inside the FOV (cells below 0.15 are not
drawn).

**From real data:** set `p_near`/`p_far` from how often the real segmentator
finds the target at different distances, `false_rate` from how many fake blobs
it outputs per frame, and the background distribution from the histogram of its
scores on background pixels.

## Known simplifications

- No occlusion.
- Background cells are independent of each other; real segmentation noise is
  smoother, in patches.
- False-positive blobs are uniform in (bearing, range), not in area, so they
  are a bit denser near the robot.
- During a failure burst the target is always missed; false positives are unchanged.
