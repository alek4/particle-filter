"""Simulator: a robot looks for a target on a table with a segmentator and a
particle filter. Left: the world. Right: live plots and status."""

import math
import random

import pygame
import pygame.gfxdraw

import theme
from particle_filter import ParticleFilter
from plot import LivePlot
from robot import Robot
from segmentator import Segmentator, render_prob_map

WIDTH, HEIGHT = 800, 600  # simulation area
PANEL_W = 300  # side panel on the right
PAD = 24  # panel inner margin
FPS = 60
SENSE_HZ = 10  # segmentator frames per second
N_PARTICLES = 100
TABLE_W, TABLE_H = 200, 120  # px, the area the target sits on


def make_table(target: tuple[float, float]) -> pygame.Rect:
    """A table containing the target at a random spot (not centred, so the prior gives no hint)."""
    left = target[0] - random.uniform(0.1, 0.9) * TABLE_W
    top = target[1] - random.uniform(0.1, 0.9) * TABLE_H
    table = pygame.Rect(int(left), int(top), TABLE_W, TABLE_H)
    return table.clamp(pygame.Rect(0, 0, WIDTH, HEIGHT))  # keep it on screen


# ----------------------------------------------------------------- drawing
def draw_grid(screen: pygame.Surface, step: int = 50) -> None:
    for x in range(step, WIDTH, step):
        pygame.draw.line(screen, theme.GRID, (x, 0), (x, HEIGHT))
    for y in range(step, HEIGHT, step):
        pygame.draw.line(screen, theme.GRID, (0, y), (WIDTH, y))


def draw_target(screen: pygame.Surface, target: tuple[float, float]) -> None:
    x, y = int(target[0]), int(target[1])
    pygame.gfxdraw.filled_circle(screen, x, y, 8, theme.BG)  # dark halo: particles never hide it
    pygame.gfxdraw.aacircle(screen, x, y, 6, theme.TARGET)
    pygame.gfxdraw.filled_circle(screen, x, y, 2, theme.TARGET)


# Legend icons, drawn centred on (x, y)
def icon_target(screen: pygame.Surface, x: int, y: int) -> None:
    pygame.gfxdraw.aacircle(screen, x, y, 5, theme.TARGET)
    pygame.gfxdraw.filled_circle(screen, x, y, 1, theme.TARGET)


def icon_estimate(screen: pygame.Surface, x: int, y: int) -> None:
    pygame.draw.rect(screen, theme.BEST, (x - 5, y - 5, 11, 11), 1)


def icon_particles(screen: pygame.Surface, x: int, y: int) -> None:
    screen.fill(theme.PARTICLE, (x - 1, y - 1, 2, 2))


def icon_map(screen: pygame.Surface, x: int, y: int) -> None:
    pygame.draw.rect(screen, theme.MAP, (x - 5, y - 4, 10, 9))


LEGEND = [("target", icon_target), ("estimate", icon_estimate), ("particles", icon_particles), ("seg. map", icon_map)]
HELP = ["L-click  move robot     R-click  move target", "WASD  drive     R  reset filter"]  # J, F: in the stats


def draw_panel(screen, fonts, plots, pf: ParticleFilter, target, following: bool) -> None:
    small, body, big = fonts
    x0, w = WIDTH + PAD, PANEL_W - 2 * PAD
    screen.fill(theme.PANEL, (WIDTH, 0, PANEL_W, HEIGHT))
    pygame.draw.line(screen, theme.BORDER, (WIDTH, 0), (WIDTH, HEIGHT))

    for plot in plots:
        plot.draw(screen, small, big)

    # Stats: label left (muted), value right
    y = 310
    rows = [
        ("Estimate", f"{pf.estimate[0]:.0f}, {pf.estimate[1]:.0f}"),
        ("Error", f"{math.dist(pf.estimate, target):.1f} px"),
        ("Jitter  (J)", pf.jitter_mode),
        ("Follow estimate  (F)", "on" if following else "off"),
    ]
    for label, value in rows:
        screen.blit(body.render(label, True, theme.MUTED), (x0, y))
        v = body.render(value, True, theme.TEXT)
        screen.blit(v, (x0 + w - v.get_width(), y))
        y += 22

    # Legend, two columns
    y += 14
    pygame.draw.line(screen, theme.BORDER, (x0, y), (x0 + w, y))
    y += 16
    for i, (label, icon) in enumerate(LEGEND):
        lx, ly = x0 + (i % 2) * (w // 2), y + (i // 2) * 22
        icon(screen, lx + 6, ly + 8)
        screen.blit(small.render(label, True, theme.MUTED), (lx + 18, ly + 2))

    # Controls, bottom
    for i, text in enumerate(HELP):
        screen.blit(small.render(text, True, theme.MUTED), (x0, HEIGHT - PAD - 34 + i * 18))


# -------------------------------------------------------------------- main
def main() -> None:
    pygame.init()
    screen = pygame.display.set_mode((WIDTH + PANEL_W, HEIGHT))
    pygame.display.set_caption("Particle Filter")
    clock = pygame.time.Clock()
    fonts = (theme.font(13), theme.font(15), theme.font(26))  # small, body, big
    bounds = (WIDTH, HEIGHT)

    robot = Robot(WIDTH / 2, HEIGHT / 2)
    target = (600.0, 200.0)  # true target position (unknown to the filter)
    segmentator = Segmentator()
    pf = ParticleFilter(N_PARTICLES, make_table(target))
    following = True  # robot follows the filter's estimate
    sense_timer = 0.0
    map_surface: pygame.Surface | None = None  # heatmap of the last map, rendered once per frame

    plot_w = PANEL_W - 2 * PAD
    entropy_plot = LivePlot(pygame.Rect(WIDTH + PAD, PAD, plot_w, 120), "Entropy", theme.ENTROPY)
    best_plot = LivePlot(pygame.Rect(WIDTH + PAD, PAD + 145, plot_w, 120), "Best cell prob.", theme.BEST)

    def reset_filter() -> None:
        pf.init_prior()
        entropy_plot.clear()
        best_plot.clear()

    running = True
    while running:
        dt = clock.tick(FPS) / 1000.0  # seconds since last frame

        # --- input
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                elif event.key == pygame.K_r:
                    reset_filter()
                elif event.key == pygame.K_j:
                    pf.jitter_mode = "fixed" if pf.jitter_mode == "adaptive" else "adaptive"
                elif event.key == pygame.K_f:
                    following = not following
                    robot.follow(pf.estimate if following else None)
            elif event.type == pygame.MOUSEBUTTONDOWN and event.pos[0] < WIDTH:  # ignore panel clicks
                if event.button == 1:  # go to the clicked point; turns following off
                    following = False
                    robot.follow(None)
                    robot.go_to(event.pos)
                elif event.button == 3:  # move the target, with a new table under it
                    target = (float(event.pos[0]), float(event.pos[1]))
                    pf.table = make_table(target)
                    reset_filter()

        robot.handle_input(pygame.key.get_pressed(), dt, bounds)
        robot.update(dt, bounds)

        # --- sense + filter, at the segmentator's rate (slower than the render loop)
        sense_timer += dt
        if sense_timer >= 1.0 / SENSE_HZ:
            sense_timer = 0.0
            pmap = segmentator.sense(robot, target)
            map_surface = render_prob_map(bounds, robot, pmap)
            pf.step(robot, pmap)
            entropy_plot.add(pf.entropy)
            best_plot.add(pf.best_prob)
            if following:
                robot.follow(pf.estimate)

        # --- draw the world. Draw order = layering: later things are on top
        screen.fill(theme.BG)
        draw_grid(screen)
        pf.draw_table(screen)
        if map_surface is not None:
            screen.blit(map_surface, (0, 0))
        pf.draw(screen)
        robot.draw(screen)
        draw_target(screen, target)

        # --- right panel (drawn last, so nothing from the world spills into it)
        draw_panel(screen, fonts, (entropy_plot, best_plot), pf, target, following)

        pygame.display.flip()

    pygame.quit()


if __name__ == "__main__":
    main()
