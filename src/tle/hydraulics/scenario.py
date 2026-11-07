"""Leak scenarios: the parameters theta of one leak situation."""
from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass, field

G = 9.81
LEAK_DISCHARGE_COEFF = 0.75


@dataclass(frozen=True)
class Scenario:
    """One leak situation.

    Attributes
    ----------
    scenario_id : unique id (stable across files)
    network : network name (``net3_prv`` or ``ltown_prv``)
    leak_node : junction id where the leak is placed
    leak_area_m2 : orifice area of the leak (log-uniform over the scenario space)
    demand_mult : global demand multiplier
    peak_shift_h : circular shift (hours) of the residential demand pattern
    pattern_amp : amplitude factor of the residential pattern around its mean
    tank_level_frac : initial level of every tank as a fraction of [min, max] level
    closed_pipes : layout edits (E3): pipes closed for this scenario
    required_pressure_m : service pressure used by the shortfall term (E2 goal change)
    energy_weight_mult : multiplier of the energy weight (E2 goal change)
    """

    scenario_id: str
    network: str
    leak_node: str
    leak_area_m2: float
    demand_mult: float = 1.0
    peak_shift_h: int = 0
    pattern_amp: float = 1.0
    tank_level_frac: float = 0.5
    closed_pipes: tuple[str, ...] = field(default_factory=tuple)
    required_pressure_m: float = 20.0
    energy_weight_mult: float = 1.0

    @property
    def emitter_coeff_lps(self) -> float:
        """EPANET emitter coefficient in (L/s)/m^0.5 for an orifice leak: Cd*A*sqrt(2g)."""
        return LEAK_DISCHARGE_COEFF * self.leak_area_m2 * math.sqrt(2 * G) * 1000.0

    def to_dict(self) -> dict:
        d = asdict(self)
        d["closed_pipes"] = list(self.closed_pipes)
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "Scenario":
        d = dict(d)
        cp = d.get("closed_pipes", ())
        if isinstance(cp, str):
            cp = tuple(p for p in cp.split(";") if p)
        d["closed_pipes"] = tuple(cp)
        fields = cls.__dataclass_fields__
        return cls(**{k: v for k, v in d.items() if k in fields})

    def content_hash(self) -> str:
        d = self.to_dict()
        d.pop("scenario_id")
        return hashlib.sha256(json.dumps(d, sort_keys=True).encode()).hexdigest()[:16]
