# Segmentator model

Code: [`segmentator.py`](../segmentator.py)

The segmentator stands in for a real single-class segmentation network. A real
network outputs one probability per pixel. This simulation outputs the same
kind of map **already projected on the ground**: a polar grid laid over the
robot's field of view (FOV), where each cell holds a value in [0, 1] that says
how much that cell looks like the target.

```
grid  : 60 bearing cells (1° each) × 50 range cells (4 px each)
        bearing = angle from the robot's heading, range = distance from the robot
```

This is the **only code that knows the true target position**. The particle
filter has its own, separate model of the segmentator (see
[particle-filter.md](particle-filter.md#update-likelihood)). That model
deliberately leaves out some things the simulator does, such as failure bursts
and the exact blob shape. If the filter assumed exactly the process that
generated the data, the test would be too optimistic.

## The model in words

Each frame starts from **background noise**: every cell gets a small random
value, almost always close to 0.

If the target is inside the FOV, it appears with a **probability that depends
on distance** (high when close, lower when far). It shows up as a **Gaussian
blob** centred on the true position plus some noise, and the blob is weaker
when the target is far.

Now and then the segmentator enters a **failure burst** lasting a few frames,
during which it does not see the target at all.

Finally, **false positives** are added, during failure bursts as well. They are
Gaussian blobs at random positions inside the FOV, and their number per frame
follows a Poisson distribution.

## Pseudocode

```
STATE
    failing ← false

FUNCTION sense(robot, target) → probability map over the FOV

    # 0. Failure bursts: the segmentator sometimes goes blind for a few frames
    IF failing:  recover with probability p_recover
    ELSE:        start failing with probability p_fail

    # 1. Background: every cell gets a small random value (mostly near 0)
    map ← low-valued noise

    # 2. True target
    IF target is inside the FOV AND not failing:
        with probability P_d(distance)                  # high when near, lower when far
            seen position ← true position + noise       # range noise grows with distance
            strength      ← random, weaker when far
            draw a blob on the map at seen position with that strength

    # 3. False positives (they still appear while failing)
    REPEAT Poisson(false_rate) times:
        draw a blob at a random place in the FOV with random strength

    RETURN map


PROCEDURE draw blob(position, strength):
    Gaussian bump centred at position, peak = strength
    its size matches the object's physical size, so it covers a narrower angle when far
    overlapping blobs keep the stronger value instead of summing
```

## Parameters

| Name | Default | Meaning |
|---|---|---|
| `p_near`, `p_far` | 0.95, 0.5 | detection probability at distance 0 and at the FOV edge; linear in between |
| `sigma_bearing` | 3° | noise on the blob centre, bearing |
| `sigma_range_frac` | 0.10 | noise on the blob centre, range, as a fraction of the distance (at least 1 px) |
| `false_rate` | 0.2 | mean number of false-positive blobs per frame |
| `p_fail` | 0.02 | per frame, chance a failure burst starts |
| `p_recover` | 0.2 | per frame, chance a burst ends (mean length: 5 frames) |
| `OBJECT_RADIUS` | 8 px | physical size of the target, which sets the blob size |
| `BACKGROUND_BETA` | (1, 15) | background values ~ Beta(1, 15), mean ≈ 0.06 |

Blob strength is `Uniform(0.6, 1.0) · (1 − 0.4 · r / R)` for the target and
`Uniform(0.4, 0.9)` for false positives.

## Calibrating from a real segmentator

- `p_near` / `p_far`: how often the real network finds the target at
  different distances.
- `false_rate`: how many spurious blobs it outputs per frame.
- Background distribution: the histogram of its scores on background pixels.

## Known simplifications

- No occlusion.
- Background cells are independent of each other. Real segmentation noise is
  smoother and comes in patches.
- False-positive blobs are uniform in (bearing, range), not in area, so they
  are slightly denser close to the robot.
- During a failure burst the target is always missed, and false positives are
  unaffected.

On screen, the map is drawn as a green heatmap inside the FOV. Cells below 0.15
are not drawn.
