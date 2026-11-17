"""Minimal scrolling line plot drawn with pygame."""

from collections import deque

import pygame

import theme


class LivePlot:
    def __init__(
        self,
        rect: pygame.Rect,
        title: str,
        color: tuple[int, int, int],
        y_min: float = 0.0,
        y_max: float = 1.0,
        max_points: int = 300,  # history length (300 points at 10 Hz = 30 s)
    ) -> None:
        self.rect: pygame.Rect = pygame.Rect(rect)
        self.title: str = title
        self.color: tuple[int, int, int] = color
        self.y_min: float = y_min
        self.y_max: float = y_max
        self.max_points: int = max_points
        self.values: deque[float] = deque(maxlen=max_points)

    def add(self, value: float) -> None:
        self.values.append(value)

    def clear(self) -> None:
        self.values.clear()

    def draw(self, screen: pygame.Surface, small: pygame.font.Font, big: pygame.font.Font) -> None:
        r = self.rect

        # Header: title on the left, current value (in the line colour) on the right
        screen.blit(small.render(self.title.upper(), True, theme.MUTED), (r.left, r.top + 6))
        value = big.render(f"{self.values[-1]:.2f}" if self.values else "–", True, self.color)
        screen.blit(value, (r.right - value.get_width(), r.top))

        # Plot area
        area = pygame.Rect(r.left, r.top + 34, r.width, r.height - 34)

        def y_of(v: float) -> float:
            return area.bottom - (v - self.y_min) / (self.y_max - self.y_min) * area.height

        # Faint gridlines at min, middle, max; small labels at the left, just under each line
        # (the bottom one just above, so it stays inside the plot)
        for v in (self.y_min, (self.y_min + self.y_max) / 2, self.y_max):
            y = y_of(v)
            pygame.draw.line(screen, theme.BORDER if v == self.y_min else theme.GRID, (area.left, y), (area.right, y))
            label = small.render(f"{v:g}", True, theme.MUTED)
            ly = y - label.get_height() - 1 if v == self.y_min else y + 2
            screen.blit(label, (area.left + 2, ly))

        # Line with a soft fill underneath. Newest point on the right.
        if len(self.values) >= 2:
            dx = area.width / (self.max_points - 1)
            pts = [(area.left + i * dx, y_of(v)) for i, v in enumerate(self.values)]
            fill = pygame.Surface(area.size, pygame.SRCALPHA)
            poly = [(pts[0][0], area.bottom), *pts, (pts[-1][0], area.bottom)]
            pygame.draw.polygon(fill, (*self.color, 28), [(x - area.left, y - area.top) for x, y in poly])
            screen.blit(fill, area.topleft)
            pygame.draw.aalines(screen, self.color, False, pts)
            pygame.draw.circle(screen, self.color, (int(pts[-1][0]), int(pts[-1][1])), 3)
