"""Simulated single-class segmentator.

Output: a probability map over the FOV, already projected on the ground: one
value per (bearing, range) cell = how much that cell looks like the target.

See docs/segmentator.md for the model in words and pseudocode.

This file is the SIMULATOR only: it is the only place that knows the true
target position. What the filter believes about the segmentator lives in
particle_filter.py.
"""

import math
from dataclasses import dataclass

import numpy as np
import pygame

import theme
from robot import Robot

MAP_BEARING_BINS = 60  # 1 deg per cell with a 60 deg FOV
MAP_RANGE_BINS = 50  # 4 px per cell with a 200 px range
OBJECT_RADIUS = 8.0  # px, physical size of the target: sets the size of its blob
BACKGROUND_BETA = (1.0, 15.0)  # background cell values ~ Beta(1, 15): mostly ~0.06


@dataclass
class ProbMap:
    """values[i, j] = how much cell (bearing bin i, range bin j) looks like the target."""

    values: np.ndarray  # (MAP_BEARING_BINS, MAP_RANGE_BINS), each in [0, 1]
    fov: float  # rad
    fov_range: float  # px

    def lookup(self, bearing: float, r: float) -> float | None:
        """Value of the cell at (bearing, range r), or None if outside the FOV."""
        if r > self.fov_range or abs(bearing) > self.fov / 2:
            return None
        nb, nr = self.values.shape
        i = min(int((bearing + self.fov / 2) / self.fov * nb), nb - 1)  # min(): the far edge
        j = min(int(r / self.fov_range * nr), nr - 1)  # belongs to the last cell
        return float(self.values[i, j])


class Segmentator:
    def __init__(
        self,
        p_near: float = 0.95,  # probability the target shows up, when very close
        p_far: float = 0.5,  # same, at the edge of the FOV range
        sigma_bearing: float = math.radians(3),  # blob centre noise, bearing
        sigma_range_frac: float = 0.10,  # blob centre noise, range (fraction of distance)
        false_rate: float = 0.2,  # mean number of false-positive blobs per frame
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

    def sense(self, robot: Robot, target: tuple[float, float]) -> ProbMap:
        """One frame: noisy background + a blob where the target is seen +
        blobs for false positives."""
        rng = self.rng
        fov, R = robot.fov, robot.fov_range
        nb, nr = MAP_BEARING_BINS, MAP_RANGE_BINS

        # 0. Failure bursts: a two-state switch (working <-> failing)
        if self.failing:
            self.failing = rng.random() >= self.p_recover
        else:
            self.failing = rng.random() < self.p_fail

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

        # 2. The real target: if visible, shows up with probability P_d(r), as a
        #    blob around a noisy centre, weaker when far
        r, b = robot.range_bearing(*target)
        visible = r <= R and abs(b) <= fov / 2
        p_detect = self.p_near + (self.p_far - self.p_near) * r / R
        if visible and not self.failing and rng.random() < p_detect:
            b_seen = b + rng.normal(0, self.sigma_bearing)
            r_seen = max(0.0, r + rng.normal(0, max(self.sigma_range_frac * r, 1.0)))
            peak = rng.uniform(0.6, 1.0) * (1.0 - 0.4 * r_seen / R)
            add_blob(b_seen, r_seen, peak)

        # 3. False positives: blobs at random places, random strength
        for _ in range(rng.poisson(self.false_rate)):
            add_blob(rng.uniform(-fov / 2, fov / 2), rng.uniform(0, R), rng.uniform(0.4, 0.9))

        return ProbMap(values=values, fov=fov, fov_range=R)


def render_prob_map(size: tuple[int, int], robot: Robot, pmap: ProbMap, threshold: float = 0.15) -> pygame.Surface:
    """Draw the map as a translucent heatmap on a new surface (render once per
    segmentator frame, then blit). robot: the pose at the time of sensing.
    Cells below `threshold` are skipped, so only blobs and the strongest
    background noise show."""
    surf = pygame.Surface(size, pygame.SRCALPHA)
    nb, nr = pmap.values.shape
    db, dr = pmap.fov / nb, pmap.fov_range / nr

    def corner(b: float, r: float) -> tuple[float, float]:
        return robot.point_at(robot.heading + b, r)

    for i, j in zip(*np.nonzero(pmap.values > threshold)):
        b0, r0 = -pmap.fov / 2 + i * db, j * dr  # the cell's first corner
        poly = [corner(b0, r0), corner(b0 + db, r0), corner(b0 + db, r0 + dr), corner(b0, r0 + dr)]
        alpha = int(230 * min(1.0, float(pmap.values[i, j])))
        pygame.draw.polygon(surf, (*theme.MAP, alpha), poly)
    return surf
