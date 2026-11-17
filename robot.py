import math

import numpy as np
import pygame
import pygame.gfxdraw

import theme


def wrap_angle(a: float) -> float:
    """Wrap an angle to [-pi, pi)."""
    return (a + math.pi) % (2 * math.pi) - math.pi


class Robot:
    """Point robot with a pose (x, y, heading) and a field of view.

    Screen coordinates: x right, y down, so heading grows clockwise on screen.
    """

    def __init__(
        self,
        x: float,
        y: float,
        heading: float = 0.0,
        speed: float = 150.0,  # px / s
        turn_rate: float = math.radians(120),  # rad / s
        fov: float = math.radians(60),  # total opening angle
        fov_range: float = 200.0,  # px
        size: float = 12.0,
    ) -> None:
        self.x = x
        self.y = y
        self.heading = heading
        self.speed = speed
        self.turn_rate = turn_rate
        self.fov = fov
        self.fov_range = fov_range
        self.size = size

        # Go-to-target controller
        self.target: tuple[float, float] | None = None
        self.arrive_tol: float = 3.0  # px: distance at which the target counts as reached
        self.k_turn: float = 4.0  # proportional gain on heading error
        self.k_dist: float = 3.0  # proportional gain on distance (slows down near target)

        # Follow controller: stay `standoff` px from a point, facing it
        self.follow_point: tuple[float, float] | None = None
        self.standoff: float = 100.0  # px; well inside the FOV range, not on top of the point
        self.manual: bool = False  # True while a drive key is pressed: it overrides the controllers

    # ------------------------------------------------------------------ motion
    def move(self, v: float, omega: float, dt: float, bounds: tuple[int, int]) -> None:
        """Unicycle model: v = forward velocity, omega = angular velocity."""
        self.heading = (self.heading + omega * dt) % (2 * math.pi)
        self.x += v * math.cos(self.heading) * dt
        self.y += v * math.sin(self.heading) * dt
        w, h = bounds
        self.x = min(max(self.x, 0), w)
        self.y = min(max(self.y, 0), h)

    def handle_input(self, keys, dt: float, bounds: tuple[int, int]) -> None:
        v = omega = 0.0
        if keys[pygame.K_w] or keys[pygame.K_UP]:
            v += self.speed
        if keys[pygame.K_s] or keys[pygame.K_DOWN]:
            v -= self.speed
        if keys[pygame.K_a] or keys[pygame.K_LEFT]:
            omega -= self.turn_rate
        if keys[pygame.K_d] or keys[pygame.K_RIGHT]:
            omega += self.turn_rate
        self.manual = bool(v or omega)
        if self.manual:
            self.target = None  # manual input cancels go-to
        self.move(v, omega, dt, bounds)

    def go_to(self, target: tuple[float, float]) -> None:
        """Set a target location; the robot drives there over the next update() calls."""
        self.target = (float(target[0]), float(target[1]))

    def follow(self, point: tuple[float, float] | None) -> None:
        """Keep `standoff` px from `point` and face it (None = stop following)."""
        self.follow_point = None if point is None else (float(point[0]), float(point[1]))

    def update(self, dt: float, bounds: tuple[int, int]) -> None:
        """Step the active controller. Priority: manual keys > go-to > follow."""
        if self.manual:
            return
        if self.target is not None:
            self._step_go_to(dt, bounds)
        elif self.follow_point is not None:
            self._step_follow(dt, bounds)

    def _step_follow(self, dt: float, bounds: tuple[int, int]) -> None:
        assert self.follow_point is not None
        dx = self.follow_point[0] - self.x
        dy = self.follow_point[1] - self.y
        dist = math.hypot(dx, dy)
        heading_err = wrap_angle(math.atan2(dy, dx) - self.heading)
        omega = max(-self.turn_rate, min(self.turn_rate, self.k_turn * heading_err))
        # Drive to close the gap to `standoff`: forward if too far, backward if too close.
        # Like go-to, only while roughly facing the point.
        v = max(-self.speed, min(self.speed, self.k_dist * (dist - self.standoff)))
        v *= max(0.0, math.cos(heading_err))
        self.move(v, omega, dt, bounds)

    def _step_go_to(self, dt: float, bounds: tuple[int, int]) -> None:
        assert self.target is not None
        dx = self.target[0] - self.x
        dy = self.target[1] - self.y
        dist = math.hypot(dx, dy)
        if dist < self.arrive_tol:
            self.target = None
            return

        heading_err = wrap_angle(math.atan2(dy, dx) - self.heading)
        omega = max(-self.turn_rate, min(self.turn_rate, self.k_turn * heading_err))
        # Only drive forward when roughly facing the target: cos(err) <= 0 means
        # turn in place; it also scales speed down while still turning.
        v = min(self.speed, self.k_dist * dist) * max(0.0, math.cos(heading_err))
        self.move(v, omega, dt, bounds)

    def in_fov(self, points: np.ndarray) -> np.ndarray:
        """points: (N, 2) array of (x, y). Returns (N,) bool: inside the FOV wedge?"""
        dx = points[:, 0] - self.x
        dy = points[:, 1] - self.y
        r = np.hypot(dx, dy)
        b = wrap_angle(np.arctan2(dy, dx) - self.heading)
        return (r <= self.fov_range) & (np.abs(b) <= self.fov / 2)

    # ----------------------------------------------------------------- drawing
    def fov_polygon(self, segments: int = 32) -> list[tuple[float, float]]:
        start = self.heading - self.fov / 2
        points = [(self.x, self.y)]
        for i in range(segments + 1):
            a = start + self.fov * i / segments
            points.append(
                (self.x + self.fov_range * math.cos(a), self.y + self.fov_range * math.sin(a))
            )
        return points

    def body_polygon(self) -> list[tuple[float, float]]:
        h, s = self.heading, self.size
        tip = (self.x + s * math.cos(h), self.y + s * math.sin(h))
        left = (self.x + s * 0.7 * math.cos(h + 2.5), self.y + s * 0.7 * math.sin(h + 2.5))
        right = (self.x + s * 0.7 * math.cos(h - 2.5), self.y + s * 0.7 * math.sin(h - 2.5))
        return [tip, left, right]

    def draw(self, screen: pygame.Surface) -> None:
        # FOV: translucent wedge on an alpha overlay
        overlay = pygame.Surface(screen.get_size(), pygame.SRCALPHA)
        fov = self.fov_polygon()
        pygame.draw.polygon(overlay, (*theme.FOV, 18), fov)
        pygame.draw.aalines(overlay, (*theme.FOV, 70), True, fov)
        screen.blit(overlay, (0, 0))

        # Go-to marker: thin line and a small ring
        if self.target is not None:
            tx, ty = int(self.target[0]), int(self.target[1])
            pygame.draw.aaline(screen, theme.BORDER, (self.x, self.y), (tx, ty))
            pygame.gfxdraw.aacircle(screen, tx, ty, 4, theme.MUTED)

        # Body
        body = self.body_polygon()
        pygame.draw.polygon(screen, theme.ROBOT, body)
        pygame.draw.aalines(screen, theme.ROBOT, True, body)
