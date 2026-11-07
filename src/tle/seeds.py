"""Seed policy: one master seed per experiment, children via SeedSequence spawn keys.

No global random generator is used anywhere in the package. Every random
stream is derived from (master seed, purpose, indices), so any run can be
re-created from its recorded spawn key alone.
"""
from __future__ import annotations

import numpy as np

PURPOSES = {
    "scenarios": 1,
    "splits": 2,
    "pairs": 3,
    "library_run": 4,
    "target_run": 5,
    "tuning": 6,
    "budget": 7,
    "models": 8,
    "resplit": 9,
    "baseline": 10,
    "bootstrap": 11,
    "reference": 14,
    "pilot": 12,
    "sensitivity": 13,
}


def seed_sequence(master: int, purpose: str, *idx: int) -> np.random.SeedSequence:
    return np.random.SeedSequence(int(master), spawn_key=(PURPOSES[purpose], *map(int, idx)))


def rng(master: int, purpose: str, *idx: int) -> np.random.Generator:
    return np.random.default_rng(seed_sequence(master, purpose, *idx))


def run_streams(ss: np.random.SeedSequence) -> tuple[np.random.Generator, np.random.Generator, np.random.Generator]:
    """(initial population, DE operators, injection noise) streams of one run seed."""
    s_init, s_de, s_inj = ss.spawn(3)
    return np.random.default_rng(s_init), np.random.default_rng(s_de), np.random.default_rng(s_inj)
