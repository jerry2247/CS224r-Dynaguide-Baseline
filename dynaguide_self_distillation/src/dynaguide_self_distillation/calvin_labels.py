from __future__ import annotations

import numpy as np


SLIDING_DOOR = 0
DRAWER = 1
SWITCH = 3
GREEN_LIGHT = 5
RED_BLOCK = slice(6, 9)
BLUE_BLOCK = slice(12, 15)
PINK_BLOCK = slice(18, 21)


def _last_frame(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values)
    if values.ndim == 3:
        return values[:, -1]
    return values


def classify_behavior(states: np.ndarray, proprios: np.ndarray) -> str:
    """Return the first CALVIN behavior expressed by a rollout."""

    states = _last_frame(states)
    proprios = _last_frame(proprios)
    if len(states) == 0:
        return "no_behavior"

    start = states[0]
    for state, proprio in zip(states, proprios):
        delta = state - start
        robot = proprio[:3]

        if np.linalg.norm(robot - state[RED_BLOCK]) < 0.1 and np.linalg.norm(delta[RED_BLOCK]) > 0.001:
            return "red_displace"
        if np.linalg.norm(robot - state[BLUE_BLOCK]) < 0.1 and np.linalg.norm(delta[BLUE_BLOCK]) > 0.001:
            return "blue_displace"
        if np.linalg.norm(robot - state[PINK_BLOCK]) < 0.1 and np.linalg.norm(delta[PINK_BLOCK]) > 0.001:
            return "pink_displace"

        if abs(delta[SLIDING_DOOR]) > 0.05:
            return "door_left" if state[SLIDING_DOOR] > start[SLIDING_DOOR] else "door_right"
        if abs(delta[DRAWER]) > 0.05:
            return "drawer_open" if state[DRAWER] > start[DRAWER] else "drawer_close"
        if abs(delta[SWITCH]) > 0.02:
            return "switch_on" if state[SWITCH] > start[SWITCH] else "switch_off"
        if abs(delta[GREEN_LIGHT]) > 0.01:
            return "button_on" if state[GREEN_LIGHT] > start[GREEN_LIGHT] else "button_off"

    return "no_behavior"
