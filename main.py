import copy
import math
import random

import pygame
import pygame.gfxdraw

import theme
from particle_filter import ParticleFilter
from plot import LivePlot
from robot import Robot
from segmentator import Segmentator, draw_detections, render_prob_map

WIDTH, HEIGHT = 800, 600  # simulation area
PANEL_W = 300  # side panel on the right
PAD = 24  # panel inner margin
FPS = 60
SENSE_HZ = 10  # segmentator frames per second
SENSOR_MODE = "map"  # "points" (blob centroids) or "map" (probability map); press M to switch
N_PARTICLES = 100
TABLE_W, TABLE_H = 200, 120  # px, the area the target sits on


def make_table(target: tuple[float, float]) -> pygame.Rect:
    """A table containing the target at a random spot (not centred, so the prior gives no hint)."""
    left = target[0] - random.uniform(0.1, 0.9) * TABLE_W
    top = target[1] - random.uniform(0.1, 0.9) * TABLE_H
    table = pygame.Rect(int(left), int(top), TABLE_W, TABLE_H)
    return table.clamp(pygame.Rect(0, 0, WIDTH, HEIGHT))  # keep it on screen


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


def draw_panel(screen, fonts, plots, pf: ParticleFilter, target, following: bool, sensor_mode: str) -> None:
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
        ("Sensor  (M)", sensor_mode),
    ]
    for label, value in rows:
        screen.blit(body.render(label, True, theme.MUTED), (x0, y))
        v = body.render(value, True, theme.TEXT)
        screen.blit(v, (x0 + w - v.get_width(), y))
        y += 22

    # Legend
    y += 14
    pygame.draw.line(screen, theme.BORDER, (x0, y), (x0 + w, y))
    y += 16
    legend = [
        ("target", lambda cx, cy: (pygame.gfxdraw.aacircle(screen, cx, cy, 5, theme.TARGET),
                                   pygame.gfxdraw.filled_circle(screen, cx, cy, 1, theme.TARGET))),
        ("estimate", lambda cx, cy: pygame.draw.rect(screen, theme.BEST, (cx - 5, cy - 5, 11, 11), 1)),
        ("particles", lambda cx, cy: screen.fill(theme.PARTICLE, (cx - 1, cy - 1, 2, 2))),
        ("detection", lambda cx, cy: pygame.gfxdraw.aacircle(screen, cx, cy, 5, theme.DET_TRUE)),
        ("false positive", lambda cx, cy: pygame.gfxdraw.aacircle(screen, cx, cy, 5, theme.DET_FALSE)),
        ("seg. map", lambda cx, cy: pygame.draw.rect(screen, theme.MAP, (cx - 5, cy - 4, 10, 9))),
    ]
    for i, (label, swatch) in enumerate(legend):
        col, row = i % 2, i // 2
        lx, ly = x0 + col * (w // 2), y + row * 22
        swatch(lx + 6, ly + 8)
        screen.blit(small.render(label, True, theme.MUTED), (lx + 18, ly + 2))

    # Controls, bottom
    help_lines = ["L-click  move robot     R-click  move target", "WASD  drive     R  reset filter"]  # J, F listed above
    for i, text in enumerate(help_lines):
        screen.blit(small.render(text, True, theme.MUTED), (x0, HEIGHT - PAD - 34 + i * 18))


def main() -> None:
    pygame.init()
    screen = pygame.display.set_mode((WIDTH + PANEL_W, HEIGHT))
    pygame.display.set_caption("Particle Filter")
    clock = pygame.time.Clock()
    fonts = (theme.font(13), theme.font(15), theme.font(26))  # small, body, big

    robot = Robot(WIDTH / 2, HEIGHT / 2)
    target = (600.0, 200.0)  # true target position (unknown to the filter)
    segmentator = Segmentator()
    pf = ParticleFilter(N_PARTICLES, make_table(target))
    detections = []
    robot_at_sense = copy.copy(robot)  # pose when the last frame was taken
    sense_timer = 0.0

    plot_w = PANEL_W - 2 * PAD
    entropy_plot = LivePlot(pygame.Rect(WIDTH + PAD, PAD, plot_w, 120), "Entropy", theme.ENTROPY)
    best_plot = LivePlot(pygame.Rect(WIDTH + PAD, PAD + 145, plot_w, 120), "Best cell prob.", theme.BEST)
    sensor_mode = SENSOR_MODE
    map_surface: pygame.Surface | None = None  # heatmap of the last map, rendered once per frame
    following = True  # robot follows the filter's estimate

    def reset_filter() -> None:
        pf.init_prior()
        entropy_plot.clear()
        best_plot.clear()

    running = True
    while running:
        dt = clock.tick(FPS) / 1000.0  # seconds since last frame

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                running = False
            elif event.type == pygame.KEYDOWN and event.key == pygame.K_r:
                reset_filter()
            elif event.type == pygame.KEYDOWN and event.key == pygame.K_j:
                pf.jitter_mode = "fixed" if pf.jitter_mode == "adaptive" else "adaptive"
            elif event.type == pygame.KEYDOWN and event.key == pygame.K_m:
                sensor_mode = "points" if sensor_mode == "map" else "map"
                detections, map_surface = [], None
            elif event.type == pygame.KEYDOWN and event.key == pygame.K_f:
                following = not following
                robot.follow(pf.estimate if following else None)
            elif event.type == pygame.MOUSEBUTTONDOWN and event.pos[0] < WIDTH:  # ignore panel clicks
                if event.button == 1:
                    following = False  # a manual destination turns following off
                    robot.follow(None)
                    robot.go_to(event.pos)
                elif event.button == 3:
                    target = (float(event.pos[0]), float(event.pos[1]))
                    pf.region = make_table(target)  # new table under the new target
                    reset_filter()

        robot.handle_input(pygame.key.get_pressed(), dt, (WIDTH, HEIGHT))
        robot.update(dt, (WIDTH, HEIGHT))

        # Segmentator runs at its own rate, slower than the render loop
        sense_timer += dt
        if sense_timer >= 1.0 / SENSE_HZ:
            sense_timer = 0.0
            robot_at_sense = copy.copy(robot)
            if sensor_mode == "map":
                pmap = segmentator.sense_map(robot, target)
                map_surface = render_prob_map((WIDTH, HEIGHT), robot_at_sense, pmap)
                pf.step(robot, pmap)
            else:
                detections = segmentator.sense(robot, target)
                pf.step(robot, detections)
            entropy_plot.add(pf.entropy)
            best_plot.add(pf.best_prob)
            if following:
                robot.follow(pf.estimate)

        # --- simulation area. Draw order = layering: later things are on top
        screen.fill(theme.BG)
        draw_grid(screen)
        pf.draw_table(screen)
        if sensor_mode == "map" and map_surface is not None:
            screen.blit(map_surface, (0, 0))
        pf.draw(screen)
        robot.draw(screen)
        if sensor_mode == "points":
            draw_detections(screen, robot_at_sense, detections)
        draw_target(screen, target)

        # --- right panel (drawn last, so nothing from the simulation spills into it)
        draw_panel(screen, fonts, (entropy_plot, best_plot), pf, target, following, sensor_mode)

        pygame.display.flip()

    pygame.quit()


if __name__ == "__main__":
    main()
