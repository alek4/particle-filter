# Segmentator model

Code: [segmentator.py](segmentator.py)

Two ways to simulate the segmentator's output; pick one with `SENSOR_MODE` in
`main.py` or press **M** in the simulator:

| Mode | Function | Output | Closer to |
|---|---|---|---|
| `points` | `sense()`     | list of point detections (bearing, range) | a pipeline that thresholds the mask and keeps blob centroids |
| `map`    | `sense_map()` | a probability per cell over the FOV       | the raw segmentation output (projected on the ground) |

- **bearing**: angle relative to the robot heading (rad).
- **range**: distance from the robot (px).

This file is the **simulator only**: it is the only code that knows the true
target position.

What the *filter* believes about the segmentator is in
[PARTICLE_FILTER.md](PARTICLE_FILTER.md) (the `likelihood()` there). It is a
*simpler* version of this model: it does not know about failure bursts. That is
on purpose: if the filter used exactly the model that generated the data, the
test would be too optimistic.

## Parameters

| Name               | Default | Meaning                                              |
|--------------------|---------|------------------------------------------------------|
| `p_near`           | 0.95    | detection probability when the target is very close  |
| `p_far`            | 0.5     | detection probability at the edge of the FOV range   |
| `sigma_bearing`    | 3°      | bearing noise (std)                                  |
| `sigma_range_frac` | 0.10    | range noise std = 10% of the distance                |
| `false_rate`       | 0.2     | average false positives per frame                    |
| `p_fail`           | 0.02    | per frame, chance a failure burst starts             |
| `p_recover`        | 0.2     | per frame, chance a burst ends (mean length 5 frames)|

Detection probability falls linearly with distance:

```
P_d(r) = p_near + (p_far - p_near) * r / fov_range
```

**From a real confusion matrix:** set `p_near`/`p_far` from how often the real
segmentator finds the target (per object, at different distances), and
`false_rate` from how many fake blobs it outputs per frame. Do not use the
per-pixel rates directly.

## Points mode: `sense(robot, target)`

Each detection is a **single point** (a blob centroid), not a blob: the
Gaussian noise only decides where the point lands.

```
# 0. failure bursts (two-state switch)
if failing:  failing stays true unless random() < p_recover
else:        failing becomes true if random() < p_fail

detections = []

# 1. the real target
r = distance(robot, target)
b = angle(robot -> target) - robot.heading
if target in FOV and not failing and random() < P_d(r):
    detections += (b + Normal(0, sigma_bearing),
                   r + Normal(0, sigma_range_frac * r))

# 2. false positives
repeat Poisson(false_rate) times:
    detections += (Uniform(-fov/2, fov/2), Uniform(0, fov_range))

return detections
```

## Map mode: `sense_map(robot, target)`

The real segmentator outputs an image with a probability per pixel. We simulate
that image **already projected on the ground**: a polar grid over the FOV,
60 bearing cells (1° each) x 50 range cells (4 px each), one value in [0, 1]
per cell.

```
# 0. failure bursts: same as points mode

# 1. background: every cell gets a small random value
values[every cell] = Beta(1, 15)          # mostly ~0.06, sometimes up to ~0.3

# 2. the real target: same "seen or missed" and centroid noise as points mode
if seen (in FOV, not failing, random() < P_d(r)):
    centre = (b + Normal(0, sigma_bearing), r + Normal(0, sigma_range_frac * r))
    peak   = Uniform(0.6, 1.0) * (1 - 0.4 * r / fov_range)    # weaker when far
    add_blob(centre, peak)

# 3. false positives: blobs at random places
repeat Poisson(false_rate) times:
    add_blob(random cell, peak = Uniform(0.4, 0.9))

add_blob(centre, peak):
    every cell = max(cell, peak * Gaussian bump around centre)
    bump width: OBJECT_RADIUS (8 px) in range, OBJECT_RADIUS / distance in bearing
                (the same object covers fewer pixels when far)
```

On screen the map is a green heatmap inside the FOV (cells below 0.15 are not
drawn).

Differences with points mode:

- the filter sees **how strong** each blob is, and weak evidence below any
  threshold is not thrown away;
- background noise is visible: random cells sometimes reach ~0.3;
- each blob has a **size** that depends on distance.

## Known simplifications

- The target is a point, so there is no occlusion and no object size.
- False positives are uniform in (bearing, range), not in area, so they are a
  bit denser near the robot.
- During a failure burst the target is always missed; false positives are unchanged.
- Map mode: background cells are independent of each other (real segmentation
  noise is smoother, in patches), and the map is not a calibrated probability.
