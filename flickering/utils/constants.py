# TODO: most of these are not used as they are for different tracking implementations

import numpy as np

CHIRALITY = 1

UNIT_VECTORS = {
    "x": np.asarray([1, 0], dtype=float),
    "y": np.asarray([0, 1], dtype=float),
    "v": np.asarray([1, 1], dtype=float) / np.sqrt(2),
    "w": np.asarray([-1, 1], dtype=float) / np.sqrt(2),
}

UNIT_VECTORS_PX = {
    "x": np.asarray([1, 0], dtype=float),
    "y": np.asarray([0, 1], dtype=float),
    "v": np.asarray([1, 1], dtype=float),
    "w": np.asarray([-1, 1], dtype=float),
}

PERPENDICULAR = {"x": "y", "y": "x", "v": "w", "w": "v"}

NEW_MAP = {"x": "r", "y": "g", "v": "m", "w": "b"}

k_B = 0.01380649

CELSIUS_TO_K = 273.15

MAX_CONTOUR_CLOSING_ITERATIONS = 1000

INITIAL_BURN_POINTS = 10

SLOPE_WINDOW = 4

HORIZONTAL_WINDOW = 5

DEFAULT_PADDING_PIXELS = 15
