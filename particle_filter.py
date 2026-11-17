"""Particle filter for the target position (x, y). Robot pose is known.

Read PARTICLE_FILTER.md for the explanation and pseudocode.
"""

import math

import numpy as np
import pygame

import theme
from robot import Robot, wrap_angle
from segmentator import Detection, ProbMap

# ---------------------------------------------------------------------------
# What the filter believes about the segmentator.
# Same values as the simulator for now, but they are independent: you can
# change one without the other to test a wrong model.

# Points mode (its "confusion matrix" + noise)
P_TP_NEAR = 0.95  # P(detected | target visible), target very close
P_TP_FAR = 0.5  # P(detected | target visible), target at the edge of the FOV range
FALSE_RATE = 0.2  # average false positives per frame
SIGMA_BEARING = math.radians(3)  # bearing noise of a true detection
SIGMA_RANGE_FRAC = 0.10  # range noise of a true detection, as fraction of distance

# Map mode: what map values look like (in practice: histograms measured on labelled data)
BACKGROUND_BETA = (1.0, 15.0)  # values on background cells ~ Beta(1, 15): mostly ~0.06
TARGET_BETA = (3.0, 2.0)  # values on the target's cell, when detected ~ Beta(3, 2): mostly ~0.6
FP_CELL_FRACTION = 0.002  # fraction of background cells covered by false-positive blobs (look like target)
M_CLIP = 0.01  # map values are clipped to [M_CLIP, 1 - M_CLIP] (avoids weights of 0 or infinity)

JITTER_MODE = "adaptive"  # "fixed" or "adaptive" (press J in the simulator to switch)
JITTER = 3.0  # px, std of the fixed jitter
JITTER_MIN = 0.2  # px, lower bound on the adaptive jitter std (so it never becomes exactly 0)
CELL = 10  # px, cell size of the grid used for entropy and the best-cell estimate
# ---------------------------------------------------------------------------


def gauss(x: float, sigma: float) -> float:
    """Normal probability density: high when x is near 0, ~0 when |x| >> sigma."""
    return math.exp(-0.5 * (x / sigma) ** 2) / (math.sqrt(2 * math.pi) * sigma)


def beta_pdf(x: float, a: float, b: float) -> float:
    """Beta probability density on [0, 1]: how typical the value x is."""
    log_norm = math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b)
    return math.exp(log_norm + (a - 1) * math.log(x) + (b - 1) * math.log(1 - x))


class ParticleFilter:
    def __init__(self, n: int, region: pygame.Rect, seed: int | None = None) -> None:
        """region: the area where the target can be (e.g. a table), used for the prior."""
        self.n: int = n
        self.region: pygame.Rect = pygame.Rect(region)
        self.rng: np.random.Generator = np.random.default_rng(seed)
        self.particles: np.ndarray = np.empty((n, 2))  # each row: a guess (x, y) of the target
        self.weights: np.ndarray = np.empty(n)  # how much we believe each guess, sums to 1
        self.jitter_mode: str = JITTER_MODE  # "fixed" or "adaptive"

        # Summary of the distribution, recomputed after every step (see compute_stats)
        self.entropy: float = 1.0  # 0 = all weight in one cell, 1 = uniform over the table
        self.best_prob: float = 0.0  # weight (probability) in the heaviest cell
        self.best_cell: pygame.Rect = pygame.Rect(0, 0, CELL, CELL)  # the heaviest cell
        self.estimate: tuple[float, float] = (0.0, 0.0)  # weighted mean inside best_cell

        self.init_prior()

    def init_prior(self) -> None:
        """Prior: the target is on the table, equally likely anywhere on it."""
        r = self.region
        self.particles = np.column_stack(
            (
                self.rng.uniform(r.left, r.right, self.n),
                self.rng.uniform(r.top, r.bottom, self.n),
            )
        )
        self.weights = np.full(self.n, 1.0 / self.n)
        self.compute_stats()

    # ------------------------------------------------------------------ filter
    def step(self, robot: Robot, measurement: list[Detection] | ProbMap) -> None:
        """One filter iteration per segmentator frame.

        measurement: point detections (points mode) or a probability map (map mode).
        """
        self.resample()
        self.jitter()
        self.update(robot, measurement)
        self.normalize()
        self.compute_stats()

    def resample(self) -> None:
        """Draw N particles from the current ones, with probability = weight."""
        idx = self.rng.choice(self.n, size=self.n, p=self.weights)
        self.particles = self.particles[idx]
        self.weights = np.full(self.n, 1.0 / self.n)

    def jitter(self) -> None:
        """Nudge each particle by a little noise, so copies made by resampling
        spread apart again. Kept on the table: the target cannot be off it."""
        if self.jitter_mode == "fixed":
            self.jitter_fixed()
        else:
            self.jitter_adaptive()
        r = self.region
        self.particles[:, 0] = np.clip(self.particles[:, 0], r.left, r.right)
        self.particles[:, 1] = np.clip(self.particles[:, 1], r.top, r.bottom)

    def jitter_fixed(self) -> None:
        """Same noise every frame: JITTER px. Simple, but puts a floor on how
        tight the cloud can get (~7 px wide with 3 px)."""
        self.particles = self.particles + self.rng.normal(0, JITTER, self.particles.shape)

    def jitter_adaptive(self) -> None:
        """Noise proportional to the current spread of the cloud (separately in
        x and y): big while particles are spread out and exploring, small once
        they have converged, so it does not blur a precise estimate.
        The factor N^(-1/6) is the standard rule of thumb for 2D (0.35 for N=500).
        """
        scale = self.n ** (-1 / 6)
        std = np.maximum(scale * self.particles.std(axis=0), JITTER_MIN)  # (std_x, std_y)
        self.particles = self.particles + self.rng.normal(0, 1, self.particles.shape) * std

    def update(self, robot: Robot, measurement: list[Detection] | ProbMap) -> None:
        for i, (px, py) in enumerate(self.particles):
            if isinstance(measurement, ProbMap):
                self.weights[i] = self.likelihood_map(px, py, robot, measurement)
            else:
                self.weights[i] = self.likelihood(px, py, robot, measurement)

    def likelihood_map(self, px: float, py: float, robot: Robot, pmap: ProbMap) -> float:
        """Map mode: how well does "the target is at (px, py)" explain the map?

        Read the map value m at the particle's cell. Same structure as points mode:

          not visible -> 1 (baseline: the map says nothing about this particle)
          visible     -> P_FN                              (target missed: cell looks like background)
                       + P_TP * f_target(m) / f_background(m)   (target detected, here)

        The ratio says how much more typical m is of a target cell than of a
        background cell: ~0 for m ~ 0.06, large for m ~ 0.5+.

        f_background includes false-positive blobs (a fraction FP_CELL_FRACTION
        of background cells looks like a target). This caps the ratio at about
        1 / FP_CELL_FRACTION: a strong value is never "impossible" on background,
        so one false blob cannot grab all the particles in one frame.
        """
        r = math.hypot(px - robot.x, py - robot.y)
        b = wrap_angle(math.atan2(py - robot.y, px - robot.x) - robot.heading)
        m = pmap.lookup(b, r)
        if m is None:
            return 1.0
        m = min(max(m, M_CLIP), 1.0 - M_CLIP)
        p_tp = P_TP_NEAR + (P_TP_FAR - P_TP_NEAR) * r / robot.fov_range
        f_target = beta_pdf(m, *TARGET_BETA)
        f_background = (1 - FP_CELL_FRACTION) * beta_pdf(m, *BACKGROUND_BETA) + FP_CELL_FRACTION * f_target
        return (1.0 - p_tp) + p_tp * f_target / f_background

    def likelihood(
        self, px: float, py: float, robot: Robot, detections: list[Detection]
    ) -> float:
        """How well does "the target is at (px, py)" explain this frame's detections?

        Only RELATIVE values matter (weights get normalised), so everything is
        measured against a baseline: a particle the robot cannot see scores 1.
        """
        # Where would the target appear to the robot if it were at this particle?
        r = math.hypot(px - robot.x, py - robot.y)
        b = wrap_angle(math.atan2(py - robot.y, px - robot.x) - robot.heading)
        visible = r <= robot.fov_range and abs(b) <= robot.fov / 2

        # Case A: not visible. The target can't have caused any detection, so
        # every detection must be a false positive. This is the baseline.
        if not visible:
            return 1.0

        # Case B: visible. Confusion matrix, but P_TP drops with distance.
        p_tp = P_TP_NEAR + (P_TP_FAR - P_TP_NEAR) * r / robot.fov_range
        p_fn = 1.0 - p_tp

        # B1: the target was missed -> all detections are false positives,
        #     same as the baseline, times the chance of missing it.
        w = p_fn

        # B2: the target was detected and it is detection z (the rest are
        #     false positives). How plausible depends on how close z is to
        #     where this particle would appear, compared to how plausible it is
        #     that z was just a false positive landing there.
        fp_density = FALSE_RATE / (robot.fov * robot.fov_range)  # false positives per (rad * px)
        sigma_r = max(SIGMA_RANGE_FRAC * r, 1.0)
        for z in detections:
            closeness = gauss(wrap_angle(z.bearing - b), SIGMA_BEARING) * gauss(z.range - r, sigma_r)
            w += p_tp * closeness / fp_density

        return w

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
        r = self.region
        nx = math.ceil(r.width / CELL)  # cells along x
        ny = math.ceil(r.height / CELL)  # cells along y

        # Which cell each particle falls in (min() keeps particles on the far edge in the last cell)
        ix = np.minimum(((self.particles[:, 0] - r.left) // CELL).astype(int), nx - 1)
        iy = np.minimum(((self.particles[:, 1] - r.top) // CELL).astype(int), ny - 1)
        cell_id = ix * ny + iy  # one number per cell
        cell_mass = np.bincount(cell_id, weights=self.weights, minlength=nx * ny)

        # Entropy, normalised by its maximum (all cells equally likely) so it is in [0, 1]
        p = cell_mass[cell_mass > 0]  # empty cells contribute 0 (0 * log 0 = 0)
        self.entropy = float(-(p * np.log(p)).sum() / math.log(nx * ny))

        # Best cell
        best = int(np.argmax(cell_mass))
        bx, by = divmod(best, ny)
        self.best_prob = float(cell_mass[best])
        self.best_cell = pygame.Rect(r.left + bx * CELL, r.top + by * CELL, CELL, CELL)
        inside = cell_id == best
        w = self.weights[inside]
        ex, ey = (w[:, None] * self.particles[inside]).sum(axis=0) / w.sum()
        self.estimate = (float(ex), float(ey))

    # ----------------------------------------------------------------- drawing
    def draw_table(self, screen: pygame.Surface) -> None:
        pygame.draw.rect(screen, theme.TABLE_FILL, self.region)
        pygame.draw.rect(screen, theme.TABLE_EDGE, self.region, width=1)

    def draw(self, screen: pygame.Surface) -> None:
        """Particles, best cell and estimate (the table is drawn by draw_table)."""
        for x, y in self.particles:
            screen.fill(theme.PARTICLE, (int(x) - 1, int(y) - 1, 2, 2))

        # Best cell (square) and the estimate inside it (crosshair with a gap)
        pygame.draw.rect(screen, theme.BEST, self.best_cell, width=1)
        ex, ey = int(self.estimate[0]), int(self.estimate[1])
        for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            pygame.draw.line(screen, theme.BEST, (ex + 3 * dx, ey + 3 * dy), (ex + 9 * dx, ey + 9 * dy), 2)
