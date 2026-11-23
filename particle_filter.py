"""Particle filter for the target position (x, y). Robot pose is known.

Read PARTICLE_FILTER.md for the explanation and pseudocode.
"""

import math

import numpy as np
import pygame

import theme
from robot import Robot
from segmentator import ProbMap

# ---------------------------------------------------------------------------
# What the filter believes about the segmentator. Independent of the simulator
# values: change one without the other to test a wrong model.
P_TP_NEAR = 0.95  # P(target shows up in the map | visible), target very close
P_TP_FAR = 0.5  # same, target at the edge of the FOV range

# What map values look like (in practice: score histograms measured on labelled data)
BACKGROUND_BETA = (1.0, 15.0)  # values on background cells ~ Beta(1, 15): mostly ~0.06
TARGET_BETA = (3.0, 2.0)  # values on the target's cell, when it shows up ~ Beta(3, 2): mostly ~0.6
FP_CELL_FRACTION = 0.002  # fraction of background cells covered by false-positive blobs
M_CLIP = 0.01  # map values are clipped to [M_CLIP, 1 - M_CLIP] (avoids weights of 0 or infinity)

JITTER_MODE = "adaptive"  # "fixed" or "adaptive" (press J in the simulator to switch)
JITTER = 3.0  # px, std of the fixed jitter
JITTER_MIN = 0.2  # px, lower bound on the adaptive jitter std (so it never becomes exactly 0)
CELL = 10  # px, cell size of the grid used for entropy and the best-cell estimate
# ---------------------------------------------------------------------------


def beta_pdf(x: float, a: float, b: float) -> float:
    """Beta probability density on [0, 1]: how typical the value x is."""
    log_norm = math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b)
    return math.exp(log_norm + (a - 1) * math.log(x) + (b - 1) * math.log(1 - x))


def f_target(m: float) -> float:
    """How typical map value m is on the target's cell."""
    return beta_pdf(m, *TARGET_BETA)


def f_background(m: float) -> float:
    """How typical map value m is on a background cell. A small fraction of
    background cells are false-positive blobs, which look like the target."""
    return (1 - FP_CELL_FRACTION) * beta_pdf(m, *BACKGROUND_BETA) + FP_CELL_FRACTION * f_target(m)


class ParticleFilter:
    def __init__(self, n: int, table: pygame.Rect, seed: int | None = None) -> None:
        """table: the area the target is known to be on (the prior)."""
        self.n = n
        self.table = pygame.Rect(table)
        self.rng = np.random.default_rng(seed)
        self.jitter_mode = JITTER_MODE  # "fixed" or "adaptive"
        self.particles = np.empty((n, 2))  # each row: a guess (x, y) of the target
        self.weights = np.empty(n)  # weights from the last update (sum to 1); stale after resample()

        # Summary of the distribution, recomputed after every step (see compute_stats)
        self.entropy = 1.0  # 0 = all weight in one cell, 1 = uniform over the table
        self.best_prob = 0.0  # weight (probability) in the heaviest cell
        self.best_cell = pygame.Rect(0, 0, CELL, CELL)  # the heaviest cell
        self.estimate = (0.0, 0.0)  # weighted mean of the particles inside best_cell

        self.init_prior()

    def init_prior(self) -> None:
        """Prior: the target is on the table, equally likely anywhere on it."""
        t = self.table
        xs = self.rng.uniform(t.left, t.right, self.n)
        ys = self.rng.uniform(t.top, t.bottom, self.n)
        self.particles = np.column_stack((xs, ys))
        self.weights = np.full(self.n, 1.0 / self.n)
        self.compute_stats()

    # ------------------------------------------------------------------ filter
    def step(self, robot: Robot, pmap: ProbMap) -> None:
        """One filter iteration per segmentator frame (Probabilistic Robotics, Table 4.3)."""
        self.jitter()
        self.update(robot, pmap)
        self.normalize()
        self.compute_stats()  # on the weighted set, before resampling
        self.resample()

    def resample(self) -> None:
        """Draw N particles from the current ones, with probability = weight.
        The result is an unweighted set: the weights are now the number of
        copies. self.weights is not used again until update() overwrites it."""
        idx = self.rng.choice(self.n, size=self.n, p=self.weights)
        self.particles = self.particles[idx]

    def jitter(self) -> None:
        """Nudge each particle by a little noise, so copies made by resampling
        spread apart again. Kept on the table: the target cannot be off it."""
        if self.jitter_mode == "fixed":
            self.jitter_fixed()
        else:
            self.jitter_adaptive()
        t = self.table
        self.particles = np.clip(self.particles, (t.left, t.top), (t.right, t.bottom))

    def jitter_fixed(self) -> None:
        """Same noise every frame: JITTER px. Simple, but puts a floor on how
        tight the cloud can get (see PARTICLE_FILTER.md, Jitter)."""
        self.particles = self.particles + self.rng.normal(0, JITTER, self.particles.shape)

    def jitter_adaptive(self) -> None:
        """Noise proportional to the current spread of the cloud (separately in
        x and y): big while particles are spread out and exploring, small once
        they have converged, so it does not blur a precise estimate.
        The factor N^(-1/6) is the standard rule of thumb for 2D (0.46 for N=100).
        """
        scale = self.n ** (-1 / 6)
        std = np.maximum(scale * self.particles.std(axis=0), JITTER_MIN)  # (std_x, std_y)
        self.particles = self.particles + self.rng.normal(0, 1, self.particles.shape) * std

    def update(self, robot: Robot, pmap: ProbMap) -> None:
        """Weights are SET (not multiplied): the particles come from a resample,
        so the old weights are already encoded in the number of copies."""
        for i, (px, py) in enumerate(self.particles):
            self.weights[i] = self.likelihood(px, py, robot, pmap)

    def likelihood(self, px: float, py: float, robot: Robot, pmap: ProbMap) -> float:
        """How much more likely is the map if the target is at (px, py) than if it is not?

        Read the map value m at the particle's cell, then:
          outside the FOV -> 1: the map says nothing about this particle
          inside the FOV  -> P(m | target here) / P(m | no target here)
        """
        r, b = robot.range_bearing(px, py)
        m = pmap.lookup(b, r)
        if m is None:
            return 1.0
        m = min(max(m, M_CLIP), 1.0 - M_CLIP)

        p_tp = P_TP_NEAR + (P_TP_FAR - P_TP_NEAR) * r / robot.fov_range  # harder to see when far
        target_here = p_tp * f_target(m) + (1 - p_tp) * f_background(m)  # seen, or missed
        no_target = f_background(m)
        return target_here / no_target

    def normalize(self) -> None:
        self.weights = self.weights / self.weights.sum()

    # ------------------------------------------------------------------- stats
    def compute_stats(self) -> None:
        """Bin the particles into CELL x CELL cells over the table, then:

        cell_mass[c] = sum of the weights of the particles in cell c
        entropy      = -sum(cell_mass * log(cell_mass)) / log(number of cells)
        best_cell    = cell with the largest mass, best_prob = that mass
        estimate     = weighted mean of the particles inside best_cell
        """
        t = self.table
        nx = math.ceil(t.width / CELL)  # cells along x
        ny = math.ceil(t.height / CELL)  # cells along y

        # Which cell each particle falls in (min() keeps particles on the far edge in the last cell)
        ix = np.minimum(((self.particles[:, 0] - t.left) // CELL).astype(int), nx - 1)
        iy = np.minimum(((self.particles[:, 1] - t.top) // CELL).astype(int), ny - 1)
        cell_id = ix * ny + iy  # one number per cell
        cell_mass = np.bincount(cell_id, weights=self.weights, minlength=nx * ny)

        # Entropy, normalised by its maximum (all cells equally likely) so it is in [0, 1]
        p = cell_mass[cell_mass > 0]  # empty cells contribute 0 (0 * log 0 = 0)
        self.entropy = float(-(p * np.log(p)).sum() / math.log(nx * ny))

        # Best cell, and the estimate inside it
        best = int(np.argmax(cell_mass))
        bx, by = divmod(best, ny)
        self.best_prob = float(cell_mass[best])
        self.best_cell = pygame.Rect(t.left + bx * CELL, t.top + by * CELL, CELL, CELL)
        inside = cell_id == best
        w = self.weights[inside]
        ex, ey = (w[:, None] * self.particles[inside]).sum(axis=0) / w.sum()
        self.estimate = (float(ex), float(ey))

    # ----------------------------------------------------------------- drawing
    def draw_table(self, screen: pygame.Surface) -> None:
        pygame.draw.rect(screen, theme.TABLE_FILL, self.table)
        pygame.draw.rect(screen, theme.TABLE_EDGE, self.table, width=1)

    def draw(self, screen: pygame.Surface) -> None:
        """Particles, best cell and estimate (the table is drawn by draw_table)."""
        for x, y in self.particles:
            screen.fill(theme.PARTICLE, (int(x) - 1, int(y) - 1, 2, 2))

        # Best cell (square) and the estimate inside it (crosshair with a gap)
        pygame.draw.rect(screen, theme.BEST, self.best_cell, width=1)
        ex, ey = int(self.estimate[0]), int(self.estimate[1])
        for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            pygame.draw.line(screen, theme.BEST, (ex + 3 * dx, ey + 3 * dy), (ex + 9 * dx, ey + 9 * dy), 2)
