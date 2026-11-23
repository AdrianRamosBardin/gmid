"""Noise parameters: naming, frequency interpolation, and expressions.

Noise depends on frequency, but the lookup table is 4-D (length x vbs x vgs x vds).
The table generator therefore simulates noise at a fixed list of frequencies and
stores every (quantity, frequency) pair as its own 4-D parameter:

    sid_<f>          total drain-current noise PSD, drain and gate AC-shorted  [A^2/Hz]
    sig_<f>          gate-current noise PSD (induced gate + gate shot noise)    [A^2/Hz]
    sid_flicker_<f>  sid_<f> - sid_thermal, clipped at 0 (f < thermal frequency) [A^2/Hz]
    sid_thermal      sid at the thermal reference frequency (white level)        [A^2/Hz]

``<f>`` is a lowercase engineering tag (`frequency_tag`): 1 -> "1hz", 1e3 -> "1khz",
1e6 -> "1meghz", 2.5e9 -> "2p5ghz", 0.1 -> "100mhz" (milli). Lowercase only, because
the generator lowercases every parameter name.

Each device entry of a table that carries noise also has a ``"noise"`` metadata dict
(see `noise_metadata`) with the simulated frequencies, the thermal reference
frequency, and the simulation temperature.

Between stored frequencies, values are interpolated log-log (exact for a power-law
spectrum such as 1/f^AF); there is no extrapolation outside the stored range.
"""

from __future__ import annotations

import math
from typing import Any, Callable

import numpy as np

from .expression import Expression

BOLTZMANN = 1.380649e-23  # J/K

SID = "sid"
SIG = "sig"
SID_FLICKER = "sid_flicker"
SID_THERMAL = "sid_thermal"
NOISE_KINDS = (SID, SIG, SID_FLICKER)

_PREFIXES = ((1e12, "t"), (1e9, "g"), (1e6, "meg"), (1e3, "k"), (1.0, ""), (1e-3, "m"))
_REL_TOL = 1e-6


# --- naming ----------------------------------------------------------------------------
def frequency_tag(frequency: float) -> str:
    """Lowercase engineering tag of a frequency: 1e3 -> '1khz', 2.5e6 -> '2p5meghz'."""
    f = float(frequency)
    if not (f > 0 and math.isfinite(f)):
        raise ValueError(f"Noise frequency must be positive and finite, got {frequency!r}.")
    for scale, prefix in _PREFIXES:
        if f >= scale * (1 - 1e-12) or scale == _PREFIXES[-1][0]:
            mantissa = f"{f / scale:.6g}"
            break
    if "e" in mantissa:  # outside the prefix range: fall back to plain scientific notation
        mantissa, prefix = f"{f:.6e}".rstrip("0").replace("e+", "e").replace("e-", "em"), ""
    return f"{mantissa.replace('.', 'p')}{prefix}hz"


def noise_parameter_name(kind: str, frequency: float) -> str:
    """Table parameter name of a frequency-dependent noise quantity, e.g. 'sid_1khz'."""
    if kind not in NOISE_KINDS:
        raise ValueError(f"Unknown noise quantity {kind!r}; expected one of {NOISE_KINDS}.")
    return f"{kind}_{frequency_tag(frequency)}"


def noise_metadata(dev: dict) -> dict | None:
    """The ``"noise"`` metadata of one device entry of a lookup table, or None."""
    meta = dev.get("noise") if isinstance(dev, dict) else None
    return meta if meta else None


def available_frequencies(meta: dict | None, kind: str, parameter_names) -> list[float]:
    """Sorted frequencies at which `kind` was actually stored (survived table cleanup)."""
    if not meta:
        return []
    names = set(parameter_names)
    return sorted(
        float(f) for f in meta.get("frequencies", []) if noise_parameter_name(kind, f) in names
    )


# --- log-log frequency interpolation ----------------------------------------------------
def _as_output(value):
    arr = np.asarray(value, dtype=float)
    return float(arr) if arr.ndim == 0 else arr


def loglog_interpolator(f0: float, f1: float, f: float) -> Callable[[Any, Any], Any]:
    """Return fn(s0, s1) giving s(f) by power-law interpolation between (f0, s0), (f1, s1).

    Falls back to linear-in-log(f) where either value is not positive (e.g. a flicker
    part clipped to zero), so the result is never NaN for valid data.
    """
    t = (math.log(f) - math.log(f0)) / (math.log(f1) - math.log(f0))

    def fn(s0, s1):
        a = np.asarray(s0, dtype=float)
        b = np.asarray(s1, dtype=float)
        pos = (a > 0) & (b > 0)
        with np.errstate(divide="ignore", invalid="ignore"):
            loglog = np.exp((1 - t) * np.log(np.where(pos, a, 1.0)) + t * np.log(np.where(pos, b, 1.0)))
        return _as_output(np.where(pos, loglog, (1 - t) * a + t * b))

    return fn


def bracket_frequency(freqs: list[float], frequency: float) -> tuple[int, int] | int:
    """Index of an exact match, or the (lo, hi) bracketing indices. Raises outside range."""
    f = float(frequency)
    for i, fi in enumerate(freqs):
        if abs(fi - f) <= _REL_TOL * fi:
            return i
    if not freqs or f < freqs[0] or f > freqs[-1]:
        span = f"[{freqs[0]:g}, {freqs[-1]:g}] Hz" if freqs else "none"
        raise ValueError(
            f"Frequency {f:g} Hz is outside the simulated noise frequencies ({span}); "
            "values are interpolated, never extrapolated. Regenerate the table with a "
            "wider noise_frequencies list."
        )
    hi = int(np.searchsorted(freqs, f))
    return hi - 1, hi


def integrate_power_law(freqs, values, f_start: float, f_stop: float) -> float:
    """Integrate a PSD sampled at `freqs` from f_start to f_stop, piecewise power-law.

    Each segment between two samples is S(f) = S_a (f/f_a)^n, integrated in closed form;
    a segment with a non-positive end point is integrated with the trapezoid rule.
    """
    freqs = np.asarray(freqs, dtype=float)
    values = np.asarray(values, dtype=float)
    if not (f_stop > f_start > 0):
        raise ValueError("Need 0 < f_start < f_stop.")
    bracket_frequency(list(freqs), f_start)  # range checks
    bracket_frequency(list(freqs), f_stop)
    # Sample points: the band edges plus every stored frequency inside the band.
    interior = freqs[(freqs > f_start * (1 + _REL_TOL)) & (freqs < f_stop * (1 - _REL_TOL))]
    pts = np.concatenate(([f_start], interior, [f_stop]))
    vals = np.array([_value_at(freqs, values, p) for p in pts])
    total = 0.0
    for fa, fb, sa, sb in zip(pts[:-1], pts[1:], vals[:-1], vals[1:]):
        if sa > 0 and sb > 0:
            n = math.log(sb / sa) / math.log(fb / fa)
            if abs(n + 1) < 1e-9:
                total += sa * fa * math.log(fb / fa)
            else:
                total += sa * fa / (n + 1) * ((fb / fa) ** (n + 1) - 1)
        else:
            total += 0.5 * (sa + sb) * (fb - fa)
    return total


def _value_at(freqs, values, f: float) -> float:
    idx = bracket_frequency(list(freqs), f)
    if isinstance(idx, int):
        return float(values[idx])
    lo, hi = idx
    return float(loglog_interpolator(freqs[lo], freqs[hi], f)(values[lo], values[hi]))


def flicker_corner(f_lo: float, s_lo: float, s_th: float, f_hi: float | None = None, s_hi: float | None = None):
    """Flicker corner frequency where the flicker part equals the thermal level.

    With one flicker sample a 1/f slope is assumed; with two the exponent AF is fitted
    (S_fl = K / f^AF). Returns NaN where the data do not define a corner.
    """
    a = np.asarray(s_lo, dtype=float)
    th = np.asarray(s_th, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        if f_hi is None:
            af = np.ones_like(a)
        else:
            b = np.asarray(s_hi, dtype=float)
            af = np.log(a / b) / math.log(f_hi / f_lo)
        fc = f_lo * (a / th) ** (1.0 / af)
    valid = (a > 0) & (th > 0) & np.isfinite(af) & (af > 0)
    return _as_output(np.where(valid, fc, np.nan))


# --- expressions ------------------------------------------------------------------------
def _latex_freq(frequency: float) -> str:
    f = float(frequency)
    for scale, prefix in ((1e12, "T"), (1e9, "G"), (1e6, "M"), (1e3, "k"), (1.0, ""), (1e-3, "m")):
        if f >= scale * (1 - 1e-12) or scale == 1e-3:
            return f"{f / scale:.4g}\\,\\mathrm{{{prefix}Hz}}"
    return f"{f:g}\\,\\mathrm{{Hz}}"


class NoiseExpressions:
    """Noise expressions for `Mosfet` and `FastMosfet`.

    The host class sets ``_noise_meta`` (the device's ``"noise"`` metadata or None) and
    ``_noise_param_names`` (the device's parameter names). All attributes are plain data,
    so hosts stay picklable for the parallel optimizer.

    Frequency-dependent quantities are methods taking a frequency in Hz; any frequency
    inside the simulated range works (log-log interpolated between stored points).
    """

    _noise_meta: dict | None = None
    _noise_param_names: list = []

    # --- introspection ---
    @property
    def has_noise(self) -> bool:
        return bool(self._noise_meta) and any(
            p == SID_THERMAL or p.startswith((SID + "_", SIG + "_")) for p in self._noise_param_names
        )

    @property
    def noise_frequencies(self) -> list[float]:
        """Frequencies at which total drain noise (sid) is stored."""
        return available_frequencies(self._noise_meta, SID, self._noise_param_names)

    def noise_parameter(self, kind: str, frequency: float) -> str:
        """Raw table parameter name of a stored noise point (for custom Expressions)."""
        self._require_noise()
        freqs = available_frequencies(self._noise_meta, kind, self._noise_param_names)
        idx = bracket_frequency(freqs, frequency)
        if not isinstance(idx, int):
            raise ValueError(
                f"{kind} is not stored at {frequency:g} Hz; stored: {freqs}. "
                f"Use the {kind}_expression(frequency) method to interpolate."
            )
        return noise_parameter_name(kind, freqs[idx])

    def _require_noise(self) -> None:
        if not self._noise_meta:
            raise ValueError(
                "This lookup table has no noise data. Generate it with "
                "SpectreSimulator(..., noise_frequencies=[...])."
            )

    def _require_param(self, name: str) -> None:
        self._require_noise()
        if name not in self._noise_param_names:
            raise ValueError(
                f"Noise parameter {name!r} is not in this lookup table (it may have been "
                "dropped because the simulator returned all zeros for it)."
            )

    def _frequency_expression(self, kind: str, frequency: float, label: str) -> Expression:
        self._require_noise()
        freqs = available_frequencies(self._noise_meta, kind, self._noise_param_names)
        if not freqs:
            raise ValueError(f"This lookup table has no {kind!r} noise data.")
        idx = bracket_frequency(freqs, frequency)
        if isinstance(idx, int):
            return Expression(variables=[noise_parameter_name(kind, freqs[idx])], label=label)
        lo, hi = idx
        return Expression(
            variables=[noise_parameter_name(kind, freqs[lo]), noise_parameter_name(kind, freqs[hi])],
            function=loglog_interpolator(freqs[lo], freqs[hi], float(frequency)),
            label=label,
        )

    @staticmethod
    def _divide_by_gm2(expr: Expression, label: str) -> Expression:
        inner = expr.function
        n = len(expr.variables)

        def fn(*args):
            gm = np.asarray(args[n], dtype=float)
            return _as_output(np.asarray(inner(*args[:n]), dtype=float) / gm**2)

        return Expression(variables=[*expr.variables, "gm"], function=fn, label=label)

    # --- drain-current noise ---
    def sid_expression(self, frequency: float) -> Expression:
        """Total drain-current noise PSD at `frequency` [A^2/Hz]."""
        return self._frequency_expression(
            SID, frequency, rf"$S_{{I_D}}\ @\ {_latex_freq(frequency)}\ (A^2/Hz)$"
        )

    def sid_flicker_expression(self, frequency: float) -> Expression:
        """Flicker part of the drain-current noise PSD at `frequency` [A^2/Hz]."""
        return self._frequency_expression(
            SID_FLICKER, frequency, rf"$S_{{I_D,fl}}\ @\ {_latex_freq(frequency)}\ (A^2/Hz)$"
        )

    @property
    def sid_thermal_expression(self) -> Expression:
        """White (thermal) drain-current noise PSD [A^2/Hz]."""
        self._require_param(SID_THERMAL)
        return Expression(variables=[SID_THERMAL], label=r"$S_{I_D,th}\ (A^2/Hz)$")

    # --- gate-current noise ---
    def sig_expression(self, frequency: float) -> Expression:
        """Gate-current noise PSD at `frequency` [A^2/Hz]."""
        return self._frequency_expression(
            SIG, frequency, rf"$S_{{I_G}}\ @\ {_latex_freq(frequency)}\ (A^2/Hz)$"
        )

    # --- input-referred (gate) voltage noise: S_vg = S_id / gm^2 ---
    def svg_expression(self, frequency: float) -> Expression:
        """Input-referred gate voltage noise PSD S_id/gm^2 at `frequency` [V^2/Hz]."""
        return self._divide_by_gm2(
            self.sid_expression(frequency), rf"$S_{{V_G}}\ @\ {_latex_freq(frequency)}\ (V^2/Hz)$"
        )

    def svg_flicker_expression(self, frequency: float) -> Expression:
        """Input-referred flicker voltage noise PSD at `frequency` [V^2/Hz]."""
        return self._divide_by_gm2(
            self.sid_flicker_expression(frequency),
            rf"$S_{{V_G,fl}}\ @\ {_latex_freq(frequency)}\ (V^2/Hz)$",
        )

    @property
    def svg_thermal_expression(self) -> Expression:
        """Input-referred thermal voltage noise PSD [V^2/Hz]."""
        return self._divide_by_gm2(self.sid_thermal_expression, r"$S_{V_G,th}\ (V^2/Hz)$")

    # --- derived figures ---
    @property
    def gamma_expression(self) -> Expression:
        """Thermal noise factor gamma = S_id,th / (4 k T gm)."""
        self._require_param(SID_THERMAL)
        kt4 = 4.0 * BOLTZMANN * (float(self._noise_meta["temperature"]) + 273.15)
        return Expression(
            variables=[SID_THERMAL, "gm"],
            function=lambda sth, gm: _as_output(np.asarray(sth, float) / (kt4 * np.abs(np.asarray(gm, float)))),
            label=r"$\gamma = S_{I_D,th} / 4kTg_m$",
        )

    @property
    def flicker_corner_expression(self) -> Expression:
        """Flicker corner frequency (flicker part == thermal level) [Hz].

        Uses the two lowest stored flicker frequencies to fit the 1/f^AF slope
        (1/f is assumed if only one is stored).
        """
        self._require_param(SID_THERMAL)
        freqs = available_frequencies(self._noise_meta, SID_FLICKER, self._noise_param_names)
        if not freqs:
            raise ValueError("No flicker noise points stored; need a frequency below the thermal one.")
        label = r"$f_{\mathrm{corner}}\ (Hz)$"
        n_lo = noise_parameter_name(SID_FLICKER, freqs[0])
        if len(freqs) == 1:
            f_lo = freqs[0]
            return Expression(
                variables=[n_lo, SID_THERMAL],
                function=lambda a, th: flicker_corner(f_lo, a, th),
                label=label,
            )
        f_lo, f_hi = freqs[0], freqs[1]
        n_hi = noise_parameter_name(SID_FLICKER, f_hi)
        return Expression(
            variables=[n_lo, n_hi, SID_THERMAL],
            function=lambda a, b, th: flicker_corner(f_lo, a, th, f_hi, b),
            label=label,
        )
