"""Forward LUT lookups for one device polarity -- the only lookup API a circuit uses.

A `DeviceTable` wraps one `~mosplot.optimizer.fast_mosfet.FastMosfet` plus a
polarity. Calling it returns a `DevicePoint` whose fields are all positive magnitudes
with conventional names, so the circuit author never writes ``vds=-abs(...)``.
The sign convention lives here and nowhere else:
NMOS has ``vds > 0``, PMOS has ``vds < 0``. For the body effect the sign of ``vbs`` follows
the table's own vbs axis: an NMOS table (and a PMOS table stored the same way) runs from 0
down, reverse bias being ``vbs = -|vsb|``; the PMOS tables of the LUT generator run from 0 up
(bulk above source), reverse bias being ``vbs = +|vsb|``.
"""

from __future__ import annotations

import math
import warnings
from dataclasses import dataclass, replace

import numpy as np

from mosplot.expressions.noise import (
    BOLTZMANN,
    SID,
    SID_FLICKER,
    SID_THERMAL,
    SIG,
    available_frequencies,
    flicker_corner,
    integrate_power_law,
    noise_parameter_name,
    _value_at,
)
from mosplot.interpolation.fast.interp import _lookup_scalar_nb


@dataclass
class DevicePoint:
    """A solved single-device operating point. All fields are positive magnitudes."""

    vgs: float  # |VGS| (= VSG for PMOS)  [V]
    vdsat: float  # |VDSAT|  [V]
    jd: float  # current density |Id| / W  [A/m]  -- always positive
    gds_id: float  # gds / |Id|  [1/V]
    cgs: float  # capacitances per the LUT device width  [F]
    cgd: float
    cdd: float
    vds_used: float  # the |VDS| this point was looked up at  [V] (for diode residuals)
    gmb_id: float = 0.0  # |gmbs| / |Id|  [1/V]; 0 when the LUT carries no gmbs data

    def small_signal(
        self, gmid: float, idd: float, width: float, table_width: float, use_gmb: bool = True
    ) -> dict:
        """Per-instance small-signal params for build_ss_model, scaled to the real width.

        gmb is a positive transconductance that boosts cascode output resistance; it is
        zero for PDKs lacking gmbs data, physically irrelevant for devices whose source sits
        at its own bulk (vsb = 0), and can be disabled with use_gmb=False (e.g. to match a
        reference model that omits the body effect).
        """
        scale = width / table_width
        return {
            "gm": gmid * idd,
            "gmb": (self.gmb_id * idd) if use_gmb else 0.0,
            "gds": self.gds_id * idd,
            "cgs": self.cgs * scale,
            "cgd": self.cgd * scale,
            "cdd": self.cdd * scale,
        }


@dataclass(frozen=True)
class NoisePoint:
    """Device noise at one operating point, as PSDs [A^2/Hz].

    `DeviceTable.noise` returns it for the LUT device width; call ``scaled(W, table_width)``
    for the real device. At fixed gm/ID, L, VDS and VSB every noise PSD is proportional to
    the width (like ID, gm and the capacitances), so scaling is exact.

    Between the stored frequencies, values are interpolated log-log (power law); outside
    them a ValueError is raised.
    """

    frequencies: tuple  # frequencies where sid is stored [Hz], ascending
    sid: tuple  # total drain-current noise PSD at each frequency
    sid_thermal: float  # white drain-current noise PSD
    flicker_frequencies: tuple  # frequencies where the flicker part is stored
    sid_flicker: tuple  # flicker drain-current noise PSD at each flicker frequency
    sig_frequencies: tuple = ()  # frequencies where gate noise is stored (empty: none)
    sig: tuple = ()  # gate-current noise PSD at each sig frequency
    temperature: float = 27.0  # simulation temperature [degC], for gamma

    def scaled(self, width: float, table_width: float) -> "NoisePoint":
        """The same operating point for a device of `width` (LUT built at `table_width`)."""
        s = width / table_width
        return replace(
            self,
            sid=tuple(v * s for v in self.sid),
            sid_thermal=self.sid_thermal * s,
            sid_flicker=tuple(v * s for v in self.sid_flicker),
            sig=tuple(v * s for v in self.sig),
        )

    # --- spot values ---
    def sid_at(self, frequency: float) -> float:
        """Total drain-current noise PSD at `frequency` [A^2/Hz]."""
        return _value_at(self.frequencies, self.sid, frequency)

    def sid_flicker_at(self, frequency: float) -> float:
        """Flicker part of the drain-current noise PSD at `frequency` [A^2/Hz]."""
        if not self.flicker_frequencies:
            raise ValueError("No flicker noise points stored.")
        return _value_at(self.flicker_frequencies, self.sid_flicker, frequency)

    def sig_at(self, frequency: float) -> float:
        """Gate-current noise PSD at `frequency` [A^2/Hz]."""
        if not self.sig_frequencies:
            raise ValueError("No gate noise (sig) in this lookup table.")
        return _value_at(self.sig_frequencies, self.sig, frequency)

    def svg_at(self, frequency: float, gm: float) -> float:
        """Input-referred gate voltage noise PSD S_id/gm^2 at `frequency` [V^2/Hz]."""
        return self.sid_at(frequency) / gm**2

    def svg_thermal(self, gm: float) -> float:
        """Input-referred thermal voltage noise PSD [V^2/Hz]."""
        return self.sid_thermal / gm**2

    # --- integrated values ---
    def integrated_sid(self, f_start: float, f_stop: float) -> float:
        """Drain-current noise power in [f_start, f_stop] [A^2] (piecewise power-law)."""
        return integrate_power_law(self.frequencies, self.sid, f_start, f_stop)

    def integrated_svg(self, f_start: float, f_stop: float, gm: float) -> float:
        """Input-referred voltage noise power in [f_start, f_stop] [V^2]."""
        return self.integrated_sid(f_start, f_stop) / gm**2

    # --- figures ---
    def gamma(self, gm: float) -> float:
        """Thermal noise factor S_id,th / (4 k T gm). Use gm of the same width."""
        return self.sid_thermal / (4.0 * BOLTZMANN * (self.temperature + 273.15) * abs(gm))

    def flicker_corner(self) -> float:
        """Frequency where the flicker part equals the thermal level [Hz] (NaN if undefined).

        Width independent. Fits 1/f^AF on the two lowest flicker points (1/f with one).
        """
        if not self.flicker_frequencies:
            return math.nan
        if len(self.flicker_frequencies) == 1:
            return flicker_corner(self.flicker_frequencies[0], self.sid_flicker[0], self.sid_thermal)
        return flicker_corner(
            self.flicker_frequencies[0], self.sid_flicker[0], self.sid_thermal,
            self.flicker_frequencies[1], self.sid_flicker[1],
        )


class DeviceTable:
    """Forward LUT lookups for one device polarity ("n" or "p")."""

    def __init__(self, fast_mosfet, polarity: str):
        assert polarity in ("n", "p"), "polarity must be 'n' or 'p'"
        self.fm = fast_mosfet
        self.pol = polarity
        self.table_width = fast_mosfet._width
        self._has_gmb = "gmbs" in getattr(fast_mosfet, "_param_names", [])
        if not self._has_gmb:
            warnings.warn(
                f"lookup table of {fast_mosfet._mos_name!r} has no gmbs: "
                "gmb = 0, so cascode output resistance is gm*ro instead of (gm+gmb)*ro "
                "(use_gmb has no effect). Regenerate the table with gmbs to include it.",
                stacklevel=2,
            )

        # --- hot-path precomputation ------------------------------------------------
        # DeviceTable.__call__ is the innermost loop of the optimizer (~1M calls/run).
        # We pay this cost once at construction so __call__ needs zero Python function
        # calls per lookup -- no expression objects, no dict lookups, no genexprs.
        #
        # Strategy:
        #   1. Build a sliced data stack containing only the 7-8 raw params that
        #      DevicePoint actually needs (out of 12 in the full LUT). The Numba JIT
        #      kernel loops over n_params, so a smaller stack means less kernel work.
        #
        #   2. Record the integer index of each DevicePoint field in that sliced stack
        #      so __call__ can compute every field with a plain array index + arithmetic.
        _tbl = fast_mosfet._table
        # vdsat may be stored as "vdsat" or "vdssat" depending on the simulator model.
        _vdsat_var = self.fm.vdsat_expression.variables[0]
        # Fixed ordered list -- deterministic indices, no set shuffling.
        _needed = ["vgs", _vdsat_var, "id", "gds", "cgs", "cgd", "cdd"]
        if self._has_gmb:
            _needed.append("gmbs")
        _idx = np.array([_tbl._param_idx[p] for p in _needed], dtype=np.int64)
        # Contiguous (n_needed, n_L, n_vbs, n_vds, n_gmid) slice of the full stack.
        self._needed_data = np.ascontiguousarray(_tbl._data_stack[_idx])
        self._lut = _tbl  # axis arrays for the JIT call
        # Integer indices into the sliced result vector, one per DevicePoint field.
        _pos = {p: i for i, p in enumerate(_needed)}
        self._i_vg = _pos["vgs"]
        self._i_vdsat = _pos[_vdsat_var]
        self._i_id = _pos["id"]
        self._i_gds = _pos["gds"]
        self._i_cgs = _pos["cgs"]
        self._i_cgd = _pos["cgd"]
        self._i_cdd = _pos["cdd"]
        self._i_gmbs = _pos.get("gmbs", 0)  # unused when _has_gmb is False
        # Precompute 1/W so jd = |id| / W becomes |id| * _inv_width (one multiply).
        self._inv_width = 1.0 / fast_mosfet._width
        # Reverse body bias is toward the side of the vbs axis that is not zero.
        self._vbs_sign = 1.0 if float(np.max(self._lut.vbs)) > 0.0 else -1.0

        # --- noise (opt-in, off the hot path) -----------------------------------------
        # Plain data only, so the parallel optimizer can copy it into its workers.
        meta = getattr(fast_mosfet, "_noise_meta", None)
        stored = [p for p in getattr(_tbl, "params", []) if p in _tbl._param_idx]
        self._noise_layout = None
        if meta and SID_THERMAL in stored:
            layout = {
                kind: available_frequencies(meta, kind, stored) for kind in (SID, SID_FLICKER, SIG)
            }
            if layout[SID]:
                layout["temperature"] = float(meta.get("temperature", 27.0))
                layout["params"] = [
                    noise_parameter_name(kind, f)
                    for kind in (SID, SID_FLICKER, SIG)
                    for f in layout[kind]
                ] + [SID_THERMAL]
                self._noise_layout = layout

    @property
    def has_noise(self) -> bool:
        return self._noise_layout is not None

    def noise(self, gmid: float, L: float, vds: float, vsb: float = 0.0) -> NoisePoint:
        """Noise of the LUT-width device at one operating point (same arguments as a call).

        Scale to the real device with ``.scaled(W, table_width)``. Not on the optimizer's
        hot path: call it only for devices whose noise a spec needs.
        """
        layout = self._noise_layout
        if layout is None:
            raise ValueError(
                f"lookup table of {self.fm._mos_name!r} has no noise data; regenerate it "
                "with SpectreSimulator(..., noise_frequencies=[...])."
            )
        vds_signed = abs(vds) if self.pol == "n" else -abs(vds)
        raw = self._lut.lookup_scalar(L, gmid, vds_signed, self._vbs_sign * abs(vsb), params=layout["params"])

        def values(kind):
            return tuple(raw[noise_parameter_name(kind, f)] for f in layout[kind])

        return NoisePoint(
            frequencies=tuple(layout[SID]),
            sid=values(SID),
            sid_thermal=raw[SID_THERMAL],
            flicker_frequencies=tuple(layout[SID_FLICKER]),
            sid_flicker=values(SID_FLICKER),
            sig_frequencies=tuple(layout[SIG]),
            sig=values(SIG),
            temperature=layout["temperature"],
        )

    def __call__(self, gmid: float, L: float, vds: float, vsb: float = 0.0) -> DevicePoint:
        """Look up one operating point. Pass positive magnitudes; signs are applied here."""
        vds_signed = abs(vds) if self.pol == "n" else -abs(vds)
        # JIT kernel on the pre-sliced 8-param stack -- no dict, no expression objects.
        raw = _lookup_scalar_nb(
            self._needed_data,
            self._lut.lengths,
            self._lut.vbs,
            self._lut.vds,
            self._lut.gmid,
            L,
            gmid,
            vds_signed,
            self._vbs_sign * abs(vsb),
        )
        # Inline all DevicePoint field computations using precomputed integer indices.
        # abs() mirrors the original semantics: every DevicePoint field is a positive
        # magnitude regardless of polarity (PMOS stores vgs/vdsat/id as negatives).
        abs_id = abs(raw[self._i_id])
        return DevicePoint(
            vgs=abs(raw[self._i_vg]),
            vdsat=abs(raw[self._i_vdsat]),
            jd=abs_id * self._inv_width,
            gds_id=raw[self._i_gds] / abs_id,
            cgs=raw[self._i_cgs],
            cgd=raw[self._i_cgd],
            cdd=raw[self._i_cdd],
            vds_used=abs(vds),
            gmb_id=abs(raw[self._i_gmbs]) / abs_id if self._has_gmb else 0.0,
        )
