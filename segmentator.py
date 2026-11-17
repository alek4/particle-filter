"""Simulated single-class segmentator. Two output types, pick one in main.py:

  - sense()     -> list of point detections (blob centroids), "points" mode
  - sense_map() -> probability map over the FOV, one value per cell, "map" mode

Read SEGMENTATOR.md for both models explained in words and pseudocode.

This file is the SIMULATOR only: it is the only place that knows the true
target position. What the filter believes about the segmentator lives in
particle_filter.py.
"""

import math
from dataclasses import dataclass

import numpy as np
import pygame
import pygame.gfxdraw

import theme
from robot import Robot, wrap_angle

# Probability map: a polar grid over the FOV, already projected on the ground
MAP_BEARING_BINS = 60  # 1 deg per cell with a 60 deg FOV
MAP_RANGE_BINS = 50  # 4 px per cell with a 200 px range
OBJECT_RADIUS = 8.0  # px, physical size of the target: sets the size of its blob
BACKGROUND_BETA = (1.0, 15.0)  # background pixel values ~ Beta(1, 15): mostly ~0.06


@dataclass
class Detection:
    bearing: float  # rad, relative to robot heading (+ = clockwise on screen)
    range: float  # px, distance from the robot
    is_false: bool  # ground truth, for drawing ONLY: the filter must never read it

    def to_world(self, robot: Robot) -> tuple[float, float]:
        a = robot.heading + self.bearing
        return robot.x + self.range * math.cos(a), robot.y + self.range * math.sin(a)


@dataclass
class ProbMap:
    """Segmentation output projected on the ground: values[i, j] = probability
    that cell (bearing bin i, range bin j) contains the target."""

    values: np.ndarray  # (MAP_BEARING_BINS, MAP_RANGE_BINS), each in [0, 1]
    fov: float  # rad
    fov_range: float  # px

    def lookup(self, bearing: float, rng: float) -> float | None:
        """Value of the cell at (bearing, range), or None if outside the FOV."""
        if rng > self.fov_range or abs(bearing) > self.fov / 2:
            return None
        nb, nr = self.values.shape
        i = min(int((bearing + self.fov / 2) / self.fov * nb), nb - 1)
        j = min(int(rng / self.fov_range * nr), nr - 1)
        return float(self.values[i, j])


class Segmentator:
    def __init__(
        self,
        p_near: float = 0.95,  # detection probability at distance 0
        p_far: float = 0.5,  # detection probability at the edge of the FOV range
        sigma_bearing: float = math.radians(3),  # bearing noise std
        sigma_range_frac: float = 0.10,  # range noise std, as a fraction of the distance
        false_rate: float = 0.2,  # mean number of false positives per frame
        p_fail: float = 0.02,  # per frame: chance a failure burst starts
        p_recover: float = 0.2,  # per frame: chance a failure burst ends
        seed: int | None = None,
    ) -> None:
        self.p_near: float = p_near
        self.p_far: float = p_far
        self.sigma_bearing: float = sigma_bearing
        self.sigma_range_frac: float = sigma_range_frac
        self.false_rate: float = false_rate
        self.p_fail: float = p_fail
        self.p_recover: float = p_recover
        self.failing: bool = False
        self.rng: np.random.Generator = np.random.default_rng(seed)

    def p_detect(self, r, fov_range: float):
        """P_d(r): linear from p_near at r=0 to p_far at r=fov_range. Works on arrays."""
        return self.p_near + (self.p_far - self.p_near) * r / fov_range

    def sigma_range(self, r):
        return np.maximum(self.sigma_range_frac * r, 1.0)  # floor avoids sigma=0 at r=0

    def _step_failure(self) -> None:
        """Failure bursts: a two-state switch (working <-> failing)."""
        if self.failing:
            self.failing = self.rng.random() >= self.p_recover
        else:
            self.failing = self.rng.random() < self.p_fail

    def _see_target(self, robot: Robot, target: tuple[float, float]) -> tuple[float, float] | None:
        """Shared by both modes: is the target seen this frame? If so, return where
        the segmentation puts it (bearing, range), with the usual noise."""
        dx, dy = target[0] - robot.x, target[1] - robot.y
        r = math.hypot(dx, dy)
        b = wrap_angle(math.atan2(dy, dx) - robot.heading)
        in_fov = r <= robot.fov_range and abs(b) <= robot.fov / 2
        if not in_fov or self.failing or self.rng.random() >= self.p_detect(r, robot.fov_range):
            return None
        return (
            b + self.rng.normal(0, self.sigma_bearing),
            max(0.0, r + self.rng.normal(0, float(self.sigma_range(r)))),
        )

    # ------------------------------------------------------------ points mode
    def sense(self, robot: Robot, target: tuple[float, float]) -> list[Detection]:
        """One frame as point detections (blob centroids)."""
        self._step_failure()
        fov, R = robot.fov, robot.fov_range
        detections: list[Detection] = []

        # 1. The real target: maybe detected, with noise
        seen = self._see_target(robot, target)
        if seen is not None:
            detections.append(Detection(bearing=seen[0], range=seen[1], is_false=False))

        # 2. False positives: Poisson count, uniform in (bearing, range) over the FOV
        for _ in range(self.rng.poisson(self.false_rate)):
            detections.append(
                Detection(
                    bearing=self.rng.uniform(-fov / 2, fov / 2),
                    range=self.rng.uniform(0, R),
                    is_false=True,
                )
            )

        return detections

    # --------------------------------------------------------------- map mode
    def sense_map(self, robot: Robot, target: tuple[float, float]) -> ProbMap:
        """One frame as a probability map: noisy background + a blob where the
        target is seen + blobs for false positives."""
        self._step_failure()
        rng = self.rng
        fov, R = robot.fov, robot.fov_range
        nb, nr = MAP_BEARING_BINS, MAP_RANGE_BINS

        # Centre of every cell, as (bearing, range) grids of shape (nb, nr)
        b_centres = -fov / 2 + (np.arange(nb) + 0.5) * fov / nb
        r_centres = (np.arange(nr) + 0.5) * R / nr
        B, Rg = np.meshgrid(b_centres, r_centres, indexing="ij")

        # 1. Background: every cell gets a small random value
        values = rng.beta(*BACKGROUND_BETA, size=(nb, nr))

        def add_blob(b0: float, r0: float, peak: float) -> None:
            """Gaussian-shaped bump of height `peak` centred at (b0, r0). Its
            angular width shrinks with distance (same object, fewer pixels)."""
            sb = max(OBJECT_RADIUS / max(r0, 1.0), fov / nb)  # rad
            sr = max(OBJECT_RADIUS, R / nr)  # px
            blob = peak * np.exp(-0.5 * (((B - b0) / sb) ** 2 + ((Rg - r0) / sr) ** 2))
            np.maximum(values, blob, out=values)

        # 2. The real target: same "seen or missed" and centroid noise as points mode.
        #    Farther away -> weaker blob.
        seen = self._see_target(robot, target)
        if seen is not None:
            peak = rng.uniform(0.6, 1.0) * (1.0 - 0.4 * seen[1] / R)
            add_blob(seen[0], seen[1], peak)

        # 3. False positives: blobs at random places, random strength
        for _ in range(rng.poisson(self.false_rate)):
            add_blob(rng.uniform(-fov / 2, fov / 2), rng.uniform(0, R), rng.uniform(0.4, 0.9))

        return ProbMap(values=values, fov=fov, fov_range=R)


# ------------------------------------------------------------------- drawing
def draw_detections(screen: pygame.Surface, robot: Robot, detections: list[Detection]) -> None:
    """robot: the robot pose at the time of sensing.

    Green = real target, red = false positive.
    """
    for det in detections:
        rgb = theme.DET_FALSE if det.is_false else theme.DET_TRUE
        x, y = det.to_world(robot)
        pygame.gfxdraw.aacircle(screen, int(x), int(y), 6, rgb)
        pygame.gfxdraw.aacircle(screen, int(x), int(y), 5, rgb)


def render_prob_map(size: tuple[int, int], robot: Robot, pmap: ProbMap, threshold: float = 0.15) -> pygame.Surface:
    """Draw the map as a translucent heatmap on a new surface (render once per
    frame of the segmentator, then blit). Cells below `threshold` are skipped,
    so only blobs and the strongest background noise show."""
    surf = pygame.Surface(size, pygame.SRCALPHA)
    nb, nr = pmap.values.shape
    db, dr = pmap.fov / nb, pmap.fov_range / nr

    def corner(b: float, r: float) -> tuple[float, float]:
        a = robot.heading + b
        return robot.x + r * math.cos(a), robot.y + r * math.sin(a)

    for i, j in zip(*np.nonzero(pmap.values > threshold)):
        b0, r0 = -pmap.fov / 2 + i * db, j * dr
        poly = [corner(b0, r0), corner(b0 + db, r0), corner(b0 + db, r0 + dr), corner(b0, r0 + dr)]
        alpha = int(230 * min(1.0, float(pmap.values[i, j])))
        pygame.draw.polygon(surf, (*theme.MAP, alpha), poly)
    return surf
