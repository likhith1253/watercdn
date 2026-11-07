"""Compiled (numba) extended-period simulation loop.

The loop calls the EPANET 2.2 toolkit functions directly through ctypes
function pointers, so a 24 h simulation needs a single Python -> native call
instead of ~1,500 ctypes calls. Semantics are those documented in
:mod:`tle.hydraulics.simulator`.
"""
from __future__ import annotations

import ctypes
import math

import numba
import numpy as np

from . import epanet as en

_LONG = np.dtype(ctypes.c_long)  # 32-bit on Windows, 64-bit on Linux/macOS
MAX_STEPS = 1024


def _bind():
    lib = en._lib()
    vp, ci = ctypes.c_void_p, ctypes.c_int
    f = {}
    for name, args in (("EN_getnodevalue", [vp, ci, ci, vp]), ("EN_getlinkvalue", [vp, ci, ci, vp]),
                       ("EN_runH", [vp, vp]), ("EN_nextH", [vp, vp])):
        fn = getattr(lib, name)
        fn.argtypes, fn.restype = args, ci
        f[name] = fn
    return f["EN_getnodevalue"], f["EN_getlinkvalue"], f["EN_runH"], f["EN_nextH"]


_getnode, _getlink, _runH, _nextH = _bind()
EN_PRESSURE, EN_DEMAND, EN_TANKVOLUME, EN_ENERGY = en.EN_PRESSURE, en.EN_DEMAND, en.EN_TANKVOLUME, en.EN_ENERGY


@numba.njit(cache=False)
def run_loop(ph, demand_idx, all_junction_idx, leak_idx, pump_idx, tank_idx, horizon_s, window_s, n_windows,
             preq, leak_coeff, tariff, detail, P, dts, times, shortfall_w, leak_w, scal, v_start, v_end, tbuf, buf):
    """Run the hydraulic time loop after EN_initH. Returns (status, n_steps).

    status: 0 ok, 1 unbalanced (non-converged step), >=100 EPANET error, -1 ended early, -2 too many steps.
    scal (out): [leak_m3, energy_kwh, energy_cost, demand_m3]
    """
    unbalanced = False
    n = 0
    for i in range(tank_idx.size):
        c = _getnode(ph, tank_idx[i], EN_TANKVOLUME, v_start[i:].ctypes.data)
        if c >= 100:
            return c, 0
    leak_m3 = 0.0
    e_kwh = 0.0
    e_cost = 0.0
    dem_m3 = 0.0
    while True:
        c = _runH(ph, tbuf[0:].ctypes.data)
        if c >= 100:
            return c, n
        if c == 1:
            unbalanced = True
        t = tbuf[0]
        c = _nextH(ph, tbuf[1:].ctypes.data)
        if c >= 100:
            return c, n
        dt = tbuf[1]
        if t >= horizon_s:
            for i in range(tank_idx.size):
                c = _getnode(ph, tank_idx[i], EN_TANKVOLUME, v_end[i:].ctypes.data)
                if c >= 100:
                    return c, n
            break
        if dt <= 0:
            return -1, n
        if n >= P.shape[0]:
            return -2, n
        dt_h = dt / 3600.0
        w = min(t // window_s, n_windows - 1)
        sf = 0.0
        for i in range(demand_idx.size):
            c = _getnode(ph, demand_idx[i], EN_PRESSURE, P[n, i:].ctypes.data)
            if c >= 100:
                return c, n
            d = preq - P[n, i]
            if d > 0.0:
                sf += d
        shortfall_w[w] += sf * dt_h
        c = _getnode(ph, leak_idx, EN_PRESSURE, buf[0:].ctypes.data)
        if c >= 100:
            return c, n
        pl = buf[0]
        q_leak = leak_coeff * math.sqrt(pl) if pl > 0.0 else 0.0
        lv = q_leak / 1000.0 * dt
        leak_m3 += lv
        leak_w[w] += lv
        e = 0.0
        for i in range(pump_idx.size):
            c = _getlink(ph, pump_idx[i], EN_ENERGY, buf[0:].ctypes.data)
            if c >= 100:
                return c, n
            e += buf[0]
        e *= dt_h
        e_kwh += e
        hour = min(t // 3600, tariff.size - 1)
        e_cost += e * tariff[hour]
        if detail:
            qa = 0.0
            for i in range(all_junction_idx.size):
                c = _getnode(ph, all_junction_idx[i], EN_DEMAND, buf[0:].ctypes.data)
                if c >= 100:
                    return c, n
                qa += buf[0]
            dem_m3 += (qa - q_leak) / 1000.0 * dt
        dts[n] = dt_h
        times[n] = t
        n += 1
    scal[0] = leak_m3
    scal[1] = e_kwh
    scal[2] = e_cost
    scal[3] = dem_m3
    return (1 if unbalanced else 0), n


@numba.njit(cache=False)
def weighted_stats(P, dts, n):
    """Time-weighted min, 5th percentile and median of the pressure matrix P[:n]."""
    m = P.shape[1]
    vals = P[:n].ravel()
    w = np.empty(n * m)
    for k in range(n):
        for i in range(m):
            w[k * m + i] = dts[k]
    order = np.argsort(vals)
    tot = w.sum()
    acc = 0.0
    p5 = vals[order[-1]]
    p50 = vals[order[-1]]
    got5 = False
    for j in range(order.size):
        acc += w[order[j]] / tot
        if not got5 and acc >= 0.05:
            p5 = vals[order[j]]
            got5 = True
        if acc >= 0.5:
            p50 = vals[order[j]]
            break
    return vals[order[0]], p5, p50


def new_buffers(n_dem: int, n_windows: int, n_tanks: int) -> dict:
    return {
        "P": np.zeros((MAX_STEPS, n_dem)), "dts": np.zeros(MAX_STEPS), "times": np.zeros(MAX_STEPS, np.int64),
        "shortfall_w": np.zeros(n_windows), "leak_w": np.zeros(n_windows), "scal": np.zeros(4),
        "v_start": np.zeros(n_tanks), "v_end": np.zeros(n_tanks), "tbuf": np.zeros(2, _LONG), "buf": np.zeros(1),
    }
