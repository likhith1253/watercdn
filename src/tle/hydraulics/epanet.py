"""Thin, fast ctypes binding to the EPANET 2.2 toolkit library bundled with WNTR.

WNTR's own ``ENepanet`` class routes every call through Python logging and
error bookkeeping, which dominates the run time of a 24 h simulation of a small
network. The optimiser needs tens of thousands of simulations per run, so this
module talks to the same shared library directly through the thread-safe
project API (``EN_*`` functions) and only implements what the study uses.

Only the EPANET binary shipped with the installed WNTR version is used, so the
simulator version is pinned by the WNTR version recorded in every run record.
"""
from __future__ import annotations

import ctypes
import os
import shutil
import sys
import tempfile
from ctypes import byref, c_char_p, c_double, c_int, c_long, c_void_p
from importlib.resources import files
from pathlib import Path

# ---- EPANET 2.2 enumerations (epanet2_enums.h) ---------------------------------
EN_NODECOUNT, EN_TANKCOUNT, EN_LINKCOUNT, EN_PATCOUNT, EN_CURVECOUNT, EN_CONTROLCOUNT = range(6)

EN_JUNCTION, EN_RESERVOIR, EN_TANK = 0, 1, 2
EN_CVPIPE, EN_PIPE, EN_PUMP, EN_PRV = 0, 1, 2, 3

# node properties
EN_ELEVATION = 0
EN_BASEDEMAND = 1
EN_PATTERN = 2
EN_EMITTER = 3
EN_TANKLEVEL = 8
EN_DEMAND = 9
EN_HEAD = 10
EN_PRESSURE = 11
EN_TANKDIAM = 17
EN_MINLEVEL = 20
EN_MAXLEVEL = 21
EN_TANKVOLUME = 24
EN_DEMANDDEFICIT = 27

# link properties
EN_DIAMETER = 0
EN_LENGTH = 1
EN_INITSTATUS = 4
EN_INITSETTING = 5
EN_FLOW = 8
EN_STATUS = 11
EN_SETTING = 12
EN_ENERGY = 13

# options
EN_EMITEXPON = 3
EN_DEMANDMULT = 4

# time parameters
EN_DURATION, EN_HYDSTEP, EN_QUALSTEP, EN_PATTERNSTEP, EN_PATTERNSTART, EN_REPORTSTEP = range(6)

# control types
EN_LOWLEVEL, EN_HILEVEL, EN_TIMER, EN_TIMEOFDAY = range(4)

# demand models
EN_DDA, EN_PDA = 0, 1

# link status values
EN_CLOSED, EN_OPEN = 0, 1


class EpanetError(RuntimeError):
    """Raised when the toolkit returns an error code (>= 100)."""

    def __init__(self, code: int, where: str):
        super().__init__(f"EPANET error {code} in {where}")
        self.code = code


def _library_path() -> str:
    if os.name == "nt":
        rel = "libepanet/windows-x64/epanet22.dll"
    elif sys.platform == "darwin":
        import platform

        rel = (
            "libepanet/darwin-arm/libepanet2.dylib"
            if "arm" in platform.platform()
            else "libepanet/darwin-x64/libepanet22.dylib"
        )
    else:
        rel = "libepanet/linux-x64/libepanet22.so"
    return str(files("wntr.epanet").joinpath(rel))


_LIB = None


def _lib():
    global _LIB
    if _LIB is None:
        path = _library_path()
        _LIB = ctypes.WinDLL(path) if os.name == "nt" else ctypes.CDLL(path)
    return _LIB


class Project:
    """One open EPANET project. Not shared between threads or processes."""

    def __init__(self, inp_file: str | Path):
        self._lib = _lib()
        self._ph = c_void_p()
        self._check(self._lib.EN_createproject(byref(self._ph)), "createproject")
        # EPANET needs real report/output file names (an empty report name sends the
        # report to stdout). They live in a private temp dir removed on close().
        self._tmpdir = tempfile.mkdtemp(prefix="tle_epanet_")
        rpt = os.path.join(self._tmpdir, "run.rpt").encode()
        out = os.path.join(self._tmpdir, "run.out").encode()
        self._check(
            self._lib.EN_open(self._ph, str(inp_file).encode(), c_char_p(rpt), c_char_p(out)),
            "open",
        )
        self._check(self._lib.EN_setstatusreport(self._ph, 0), "setstatusreport")
        self._hyd_open = False
        # cache bound functions (avoids attribute lookups in the inner loop)
        self._getnodevalue = self._lib.EN_getnodevalue
        self._getlinkvalue = self._lib.EN_getlinkvalue
        self._dbl = c_double()

    # ---- error handling -------------------------------------------------------
    @staticmethod
    def _check(code: int, where: str) -> int:
        if code >= 100:
            raise EpanetError(code, where)
        return code  # 0 = ok, 1..6 = warnings (returned to caller)

    def close(self) -> None:
        if self._ph is not None and self._ph.value:
            if self._hyd_open:
                self._lib.EN_closeH(self._ph)
                self._hyd_open = False
            self._lib.EN_close(self._ph)
            self._lib.EN_deleteproject(self._ph)
            self._ph = None
            shutil.rmtree(self._tmpdir, ignore_errors=True)

    def __del__(self):  # pragma: no cover - best effort cleanup
        try:
            self.close()
        except Exception:
            pass

    # ---- counts and ids -------------------------------------------------------
    def count(self, what: int) -> int:
        v = c_int()
        self._check(self._lib.EN_getcount(self._ph, what, byref(v)), "getcount")
        return v.value

    def node_index(self, node_id: str) -> int:
        v = c_int()
        self._check(self._lib.EN_getnodeindex(self._ph, node_id.encode(), byref(v)), f"getnodeindex {node_id}")
        return v.value

    def link_index(self, link_id: str) -> int:
        v = c_int()
        self._check(self._lib.EN_getlinkindex(self._ph, link_id.encode(), byref(v)), f"getlinkindex {link_id}")
        return v.value

    def node_id(self, index: int) -> str:
        buf = ctypes.create_string_buffer(64)
        self._check(self._lib.EN_getnodeid(self._ph, index, buf), "getnodeid")
        return buf.value.decode()

    def link_id(self, index: int) -> str:
        buf = ctypes.create_string_buffer(64)
        self._check(self._lib.EN_getlinkid(self._ph, index, buf), "getlinkid")
        return buf.value.decode()

    def node_type(self, index: int) -> int:
        v = c_int()
        self._check(self._lib.EN_getnodetype(self._ph, index, byref(v)), "getnodetype")
        return v.value

    def link_type(self, index: int) -> int:
        v = c_int()
        self._check(self._lib.EN_getlinktype(self._ph, index, byref(v)), "getlinktype")
        return v.value

    def link_nodes(self, index: int) -> tuple[int, int]:
        a, b = c_int(), c_int()
        self._check(self._lib.EN_getlinknodes(self._ph, index, byref(a), byref(b)), "getlinknodes")
        return a.value, b.value

    # ---- values ----------------------------------------------------------------
    def get_node(self, index: int, prop: int) -> float:
        d = self._dbl
        code = self._getnodevalue(self._ph, index, prop, byref(d))
        if code >= 100:
            raise EpanetError(code, "getnodevalue")
        return d.value

    def set_node(self, index: int, prop: int, value: float) -> None:
        self._check(self._lib.EN_setnodevalue(self._ph, index, prop, c_double(value)), "setnodevalue")

    def get_link(self, index: int, prop: int) -> float:
        d = self._dbl
        code = self._getlinkvalue(self._ph, index, prop, byref(d))
        if code >= 100:
            raise EpanetError(code, "getlinkvalue")
        return d.value

    def set_link(self, index: int, prop: int, value: float) -> None:
        self._check(self._lib.EN_setlinkvalue(self._ph, index, prop, c_double(value)), "setlinkvalue")

    def pattern_index(self, pat_id: str) -> int:
        v = c_int()
        self._check(self._lib.EN_getpatternindex(self._ph, pat_id.encode(), byref(v)), "getpatternindex")
        return v.value

    def pattern_len(self, index: int) -> int:
        v = c_int()
        self._check(self._lib.EN_getpatternlen(self._ph, index, byref(v)), "getpatternlen")
        return v.value

    def get_pattern_value(self, index: int, period: int) -> float:
        d = c_double()
        self._check(self._lib.EN_getpatternvalue(self._ph, index, period, byref(d)), "getpatternvalue")
        return d.value

    def set_pattern_value(self, index: int, period: int, value: float) -> None:
        self._check(self._lib.EN_setpatternvalue(self._ph, index, period, c_double(value)), "setpatternvalue")

    def set_option(self, option: int, value: float) -> None:
        self._check(self._lib.EN_setoption(self._ph, option, c_double(value)), "setoption")

    def get_option(self, option: int) -> float:
        d = c_double()
        self._check(self._lib.EN_getoption(self._ph, option, byref(d)), "getoption")
        return d.value

    def set_time_param(self, param: int, value: int) -> None:
        self._check(self._lib.EN_settimeparam(self._ph, param, c_long(int(value))), "settimeparam")

    def get_time_param(self, param: int) -> int:
        v = c_long()
        self._check(self._lib.EN_gettimeparam(self._ph, param, byref(v)), "gettimeparam")
        return v.value

    def set_demand_model(self, model: int, pmin: float, preq: float, pexp: float) -> None:
        self._check(
            self._lib.EN_setdemandmodel(self._ph, model, c_double(pmin), c_double(preq), c_double(pexp)),
            "setdemandmodel",
        )

    # ---- controls --------------------------------------------------------------
    def add_control(self, ctype: int, link: int, setting: float, node: int, level: float) -> int:
        idx = c_int()
        self._check(
            self._lib.EN_addcontrol(self._ph, ctype, link, c_double(setting), node, c_double(level), byref(idx)),
            "addcontrol",
        )
        return idx.value

    def set_control(self, index: int, ctype: int, link: int, setting: float, node: int, level: float) -> None:
        self._check(
            self._lib.EN_setcontrol(self._ph, index, ctype, link, c_double(setting), node, c_double(level)),
            "setcontrol",
        )

    def delete_all_controls(self) -> None:
        for i in range(self.count(EN_CONTROLCOUNT), 0, -1):
            self._check(self._lib.EN_deletecontrol(self._ph, i), "deletecontrol")

    # ---- hydraulic run ---------------------------------------------------------
    def open_h(self) -> None:
        if not self._hyd_open:
            self._check(self._lib.EN_openH(self._ph), "openH")
            self._hyd_open = True

    def init_h(self) -> None:
        # EN_INITFLOW (10): re-initialise link flows so every run is independent of
        # the previously simulated decision (otherwise results differ at ~1e-8).
        self._check(self._lib.EN_initH(self._ph, 10), "initH")

    def run_h(self) -> tuple[int, int]:
        """Solve the current time step. Returns (time in s, warning code; 0 = none)."""
        t = c_long()
        code = self._check(self._lib.EN_runH(self._ph, byref(t)), "runH")
        return t.value, code

    def next_h(self) -> int:
        dt = c_long()
        self._check(self._lib.EN_nextH(self._ph, byref(dt)), "nextH")
        return dt.value

    def close_h(self) -> None:
        if self._hyd_open:
            self._lib.EN_closeH(self._ph)
            self._hyd_open = False

    def save_inp(self, path: str | Path) -> None:
        self._check(self._lib.EN_saveinpfile(self._ph, str(path).encode()), "saveinpfile")
