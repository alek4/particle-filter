"""Point robot with a pose (x, y, heading), a field of view and three ways to move:
manual keys, go-to a clicked point, or follow a point (the filter's estimate)."""

import math

import pygame
import pygame.gfxdraw

import theme


def wrap_angle(a: float) -> float:
    """Wrap an angle to [-pi, pi)."""
    return (a + math.pi) % (2 * math.pi) - math.pi


class Robot:
    """Screen coordinates: x right, y down, so heading grows clockwise on screen."""

    def __init__(
        self,
        x: float,
        y: float,
        heading: float = 0.0,
        speed: float = 150.0,  # px / s
        turn_rate: float = math.radians(120),  # rad / s
        fov: float = math.radians(60),  # total opening angle
        fov_range: float = 200.0,  # px
        size: float = 12.0,  # px, drawing only
    ) -> None:
        self.x = x
        self.y = y
        self.heading = heading
        self.speed = speed
        self.turn_rate = turn_rate
        self.fov = fov
        self.fov_range = fov_range
        self.size = size

        self.k_turn = 4.0  # proportional gain on heading error
        self.k_dist = 3.0  # proportional gain on distance (slows down near the point)

        self.manual = False  # True while a drive key is pressed: overrides go-to and follow
        self.goal: tuple[float, float] | None = None  # go-to destination
        self.arrive_tol = 3.0  # px: distance at which the goal counts as reached
        self.follow_point: tuple[float, float] | None = None
        self.standoff = 100.0  # px kept from the follow point: well inside the FOV range

    def range_bearing(self, x: float, y: float) -> tuple[float, float]:
        """Distance to (x, y), and its angle relative to the heading (+ = clockwise)."""
        dx, dy = x - self.x, y - self.y
        return math.hypot(dx, dy), wrap_angle(math.atan2(dy, dx) - self.heading)

    # ------------------------------------------------------------------ motion
    def move(self, v: float, omega: float, dt: float, bounds: tuple[int, int]) -> None:
        """Unicycle model: v = forward velocity, omega = angular velocity. Stays inside bounds."""
        self.heading = (self.heading + omega * dt) % (2 * math.pi)
        self.x += v * math.cos(self.heading) * dt
        self.y += v * math.sin(self.heading) * dt
        w, h = bounds
        self.x = min(max(self.x, 0), w)
        self.y = min(max(self.y, 0), h)

    def handle_input(self, keys, dt: float, bounds: tuple[int, int]) -> None:
        """WASD / arrows. Any drive key cancels go-to."""
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
            self.goal = None
        self.move(v, omega, dt, bounds)

    def go_to(self, goal: tuple[float, float]) -> None:
        """Drive to `goal` over the next update() calls, then stop."""
        self.goal = (float(goal[0]), float(goal[1]))

    def follow(self, point: tuple[float, float] | None) -> None:
        """Keep `standoff` px from `point`, facing it (None = stop following)."""
        self.follow_point = None if point is None else (float(point[0]), float(point[1]))

    def update(self, dt: float, bounds: tuple[int, int]) -> None:
        """Step the active controller. Priority: manual keys > go-to > follow."""
        if self.manual:
            return
        if self.goal is not None:
            self._step_go_to(self.goal, dt, bounds)
        elif self.follow_point is not None:
            self._step_follow(self.follow_point, dt, bounds)

    def _steer(self, point: tuple[float, float]) -> tuple[float, float, float]:
        """Distance to `point`, heading error towards it, and the turn rate that
        reduces that error (proportional, capped at turn_rate)."""
        dx, dy = point[0] - self.x, point[1] - self.y
        heading_err = wrap_angle(math.atan2(dy, dx) - self.heading)
        omega = max(-self.turn_rate, min(self.turn_rate, self.k_turn * heading_err))
        return math.hypot(dx, dy), heading_err, omega

    def _step_go_to(self, goal: tuple[float, float], dt: float, bounds: tuple[int, int]) -> None:
        dist, heading_err, omega = self._steer(goal)
        if dist < self.arrive_tol:
            self.goal = None
            return
        # Drive forward only when roughly facing the goal: cos(err) <= 0 means
        # turn in place; it also scales speed down while still turning.
        v = min(self.speed, self.k_dist * dist) * max(0.0, math.cos(heading_err))
        self.move(v, omega, dt, bounds)

    def _step_follow(self, point: tuple[float, float], dt: float, bounds: tuple[int, int]) -> None:
        dist, heading_err, omega = self._steer(point)
        # Close the gap to `standoff`: forward if too far, backward if too close,
        # only while roughly facing the point (as in go-to).
        v = max(-self.speed, min(self.speed, self.k_dist * (dist - self.standoff)))
        v *= max(0.0, math.cos(heading_err))
        self.move(v, omega, dt, bounds)

    # ----------------------------------------------------------------- drawing
    def point_at(self, angle: float, dist: float) -> tuple[float, float]:
        """The point `dist` px away from the robot in absolute direction `angle`."""
        return self.x + dist * math.cos(angle), self.y + dist * math.sin(angle)

    def fov_polygon(self, segments: int = 32) -> list[tuple[float, float]]:
        start = self.heading - self.fov / 2
        arc = [self.point_at(start + self.fov * i / segments, self.fov_range) for i in range(segments + 1)]
        return [(self.x, self.y), *arc]

    def body_polygon(self) -> list[tuple[float, float]]:
        h, s = self.heading, self.size
        return [self.point_at(h, s), self.point_at(h + 2.5, s * 0.7), self.point_at(h - 2.5, s * 0.7)]

    def draw(self, screen: pygame.Surface) -> None:
        # FOV: translucent wedge on an alpha overlay
        overlay = pygame.Surface(screen.get_size(), pygame.SRCALPHA)
        fov = self.fov_polygon()
        pygame.draw.polygon(overlay, (*theme.FOV, 18), fov)
        pygame.draw.aalines(overlay, (*theme.FOV, 70), True, fov)
        screen.blit(overlay, (0, 0))

        # Go-to goal: thin line and a small ring
        if self.goal is not None:
            gx, gy = int(self.goal[0]), int(self.goal[1])
            pygame.draw.aaline(screen, theme.BORDER, (self.x, self.y), (gx, gy))
            pygame.gfxdraw.aacircle(screen, gx, gy, 4, theme.MUTED)

        # Body
        body = self.body_polygon()
        pygame.draw.polygon(screen, theme.ROBOT, body)
        pygame.draw.aalines(screen, theme.ROBOT, True, body)
