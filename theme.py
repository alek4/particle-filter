"""Colours and fonts shared by every drawing function."""

import pygame

BG = (14, 15, 19)
GRID = (22, 24, 30)
PANEL = (19, 21, 27)
BORDER = (40, 44, 54)
TEXT = (230, 232, 236)
MUTED = (120, 126, 138)

ROBOT = (235, 238, 245)
FOV = (170, 180, 255)  # drawn translucent
TABLE_FILL = (20, 22, 28)
TABLE_EDGE = (58, 63, 76)
PARTICLE = (157, 140, 255)
TARGET = (255, 255, 255)
MAP = (91, 228, 155)  # segmentation map heatmap (drawn translucent)

ENTROPY = (245, 185, 66)  # amber
BEST = (76, 201, 240)  # cyan: best-cell plot, best cell and estimate on the map


def font(size: int) -> pygame.font.Font:
    """A clean sans font if installed, otherwise pygame's built-in one."""
    return pygame.font.SysFont("liberationsans,freesans,arial", size)
