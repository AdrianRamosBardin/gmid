# imports <<<
import os
import tempfile
import warnings
from dataclasses import dataclass, field

import numpy as np

from mosplot.expressions.noise import (
    SID,
    SID_FLICKER,
    SID_THERMAL,
    SIG,
    noise_parameter_name,
)

from .base_simulator import BaseSimulator
from .parsers.spectre import parse_file
# >>>

# Analysis names used in the netlist; the parser uses them to tell blocks apart.
_NOISE_DRAIN = "noise_id"
_NOISE_GATE = "noise_ig"
_NOISE_MARKER = "__noise__"  # parameter_table value of a noise parameter


@dataclass
class SpectreAnalysis:
    """Parsed Spectre output of one job, split into DC and noise plot blocks."""

    dc: list = field(default_factory=list)  # structured arrays, one per outer VGS point
    noise: list = field(default_factory=list)  # (plot metadata, structured array) pairs


class SpectreSimulator(BaseSimulator):
    """Spectre backend.

    Noise
    -----
    Pass ``noise_frequencies`` (Hz) to also run a Spectre ``noise`` analysis at every
    (VGS, VDS) point of the sweep. Both AC terminals are held by ideal voltage sources,
    so the noise current through them is the short-circuit noise of the device:

    * ``oprobe=VDS`` -> total drain-current noise PSD   -> ``sid_<f>``  [A^2/Hz]
    * ``oprobe=VGS`` -> gate-current noise PSD          -> ``sig_<f>``  [A^2/Hz]
      (only with ``gate_noise=True``)

    The thermal/flicker split is derived from the frequency points: the drain noise at
    ``noise_thermal_frequency`` (default: the highest frequency) is taken as the white
    level ``sid_thermal``; ``sid_flicker_<f> = max(sid_<f> - sid_thermal, 0)`` for every
    lower frequency. Choose the thermal frequency well above the flicker corner of every
    swept bias point.

    Spectre reports the noise output as a root spectral density (A/sqrt(Hz)); it is
    squared into a PSD. Set ``NOISE_OUTPUT_IS_ROOT_PSD = False`` on the class if your
    Spectre version already writes A^2/Hz.
    """

    NOISE_OUTPUT_IS_ROOT_PSD = True

    def __init__(
        self,
        *,
        temperature=27,
        raw_spice=None,
        lib_mappings=None,
        include_paths=None,
        simulator_path="spectre",
        mos_spice_symbols=("M1", "M1"),
        device_parameters=None,
        parameters_to_save=None,
        noise_frequencies=None,
        noise_thermal_frequency=None,
        gate_noise=True,
    ):
        params = list(parameters_to_save) if parameters_to_save is not None else [
            "id", "vth", "vdsat", "gm", "gmbs", "gds", "cgg", "cgs", "cgb", "cgd", "cdd"
        ]

        self.noise_frequencies, self.noise_thermal_frequency = self._normalize_noise_frequencies(
            noise_frequencies, noise_thermal_frequency
        )
        self.gate_noise = bool(gate_noise)
        # The noise columns are regular parameters, so the generic generator allocates,
        # fills, cleans and saves them like any other. Dedupe: clone() passes them back in.
        params += [p for p in self.noise_parameter_names() if p not in params]

        super().__init__(
            raw_spice=raw_spice,
            temperature=temperature,
            lib_mappings=lib_mappings,
            include_paths=include_paths,
            simulator_path=simulator_path,
            mos_spice_symbols=mos_spice_symbols,
            device_parameters=device_parameters if device_parameters is not None else {"w": 10e-6},
            parameters_to_save=params,
        )
        self._init_config.update(
            noise_frequencies=noise_frequencies,
            noise_thermal_frequency=noise_thermal_frequency,
            gate_noise=gate_noise,
        )

    # --- noise configuration --------------------------------------------------------------
    @staticmethod
    def _normalize_noise_frequencies(frequencies, thermal_frequency):
        if not frequencies:
            if thermal_frequency is not None:
                raise ValueError("noise_thermal_frequency requires noise_frequencies.")
            return [], None
        freqs = sorted({float(f) for f in frequencies})
        if any(not (f > 0 and np.isfinite(f)) for f in freqs):
            raise ValueError(f"noise_frequencies must be positive and finite, got {frequencies}.")
        f_th = freqs[-1] if thermal_frequency is None else float(thermal_frequency)
        if not any(np.isclose(f_th, f, rtol=1e-9, atol=0) for f in freqs):
            freqs = sorted(freqs + [f_th])
        # All parameter names must be distinct after formatting.
        tags = [noise_parameter_name(SID, f) for f in freqs]
        if len(set(tags)) != len(tags):
            raise ValueError(f"noise_frequencies {freqs} are too close to be told apart.")
        return freqs, f_th

    @property
    def noise_enabled(self):
        return bool(self.noise_frequencies)

    def _flicker_frequencies(self):
        return [f for f in self.noise_frequencies if f < self.noise_thermal_frequency * (1 - 1e-9)]

    def noise_parameter_names(self):
        """Every table parameter the noise analysis produces, in a fixed order."""
        if not self.noise_enabled:
            return []
        names = [noise_parameter_name(SID, f) for f in self.noise_frequencies]
        names.append(SID_THERMAL)
        names += [noise_parameter_name(SID_FLICKER, f) for f in self._flicker_frequencies()]
        if self.gate_noise:
            names += [noise_parameter_name(SIG, f) for f in self.noise_frequencies]
        return names

    def noise_metadata(self):
        """Stored as lookup_table[model]["noise"]; None when noise is off."""
        if not self.noise_enabled:
            return None
        return {
            "frequencies": list(self.noise_frequencies),
            "thermal_frequency": self.noise_thermal_frequency,
            "temperature": float(self.temperature),
            "gate_noise": self.gate_noise,
            "method": "spectre noise analysis, drain/gate AC-shorted; thermal = sid at thermal_frequency",
        }

    # --- simulator plumbing ---------------------------------------------------------------
    def make_temp_files(self):
        self.tmp_dir = tempfile.mkdtemp()
        self.input_file_path = os.path.join(self.tmp_dir, "input.scs")
        self.log_file_path = os.path.join(self.tmp_dir, "spectre.log")
        self.output_file_path = os.path.join(self.tmp_dir, "output.raw")

    def build_simulation_command(self, verbose):
        if verbose:
            return [self.simulator_path, "-format", "nutbin", "-raw", self.output_file_path, self.input_file_path]
        return [self.simulator_path, "-format", "nutbin", "-raw", self.output_file_path, "=log", self.log_file_path, self.input_file_path]

    def setup_op_simulation(self, sweep):
        symbol = self.mos_spice_symbols[1]
        return [
            f"o_temp options temp={self.temperature} tnom={self.temperature}",
            f"save {symbol}:oppoint",
            "dcop dc oppoint=screen",
        ]

    def setup_dc_simulation(self, sweep):
        symbol = self.mos_spice_symbols[1]
        id_terminal = "p" if sweep.mos_type == "pmos" else "n"
        id_probe = f"VDS:{id_terminal}"

        self.parameter_table = {
            "id":    id_probe,
            "vth":   f"{symbol}:vth",
            "vdsat": (f"{symbol}:vdsat", f"{symbol}:vdss"),  # BSIM: vdsat; PSP: vdss
            "gm":    f"{symbol}:gm",
            "gmbs":  (f"{symbol}:gmbs", f"{symbol}:gmb"),  # BSIM: gmbs; PSP: gmb
            "gds":   f"{symbol}:gds",
            "cgg":   f"{symbol}:cgg",
            "cgs":   f"{symbol}:cgs",
            "cgd":   f"{symbol}:cgd",
            "cgb":   f"{symbol}:cgb",
            "cdd":   f"{symbol}:cdd",
            "css":   f"{symbol}:css",
        }
        self.parameter_table.update({p: _NOISE_MARKER for p in self.noise_parameter_names()})

        self.parameter_table = self.select_parameters(self.parameter_table)

        vgs_start, vgs_stop, vgs_step = sweep.vgs
        vds_start, vds_stop, vds_step = sweep.vds

        lines = [
            f"o_temp options temp={self.temperature} tnom={self.temperature}",
            f"save {symbol}:oppoint {id_probe}",
            f"vgs_swp sweep param=dc dev=VGS start={vgs_start} stop={vgs_stop} step={vgs_step} {{",
            f"    vds_dc dc param=dc dev=VDS start={vds_start} stop={vds_stop} step={vds_step}",
            "}",
        ]
        if self._noise_requested():
            values = " ".join(f"{f:.12g}" for f in self.noise_frequencies)
            lines += [
                f"nswp_vgs sweep param=dc dev=VGS start={vgs_start} stop={vgs_stop} step={vgs_step} {{",
                f"    nswp_vds sweep param=dc dev=VDS start={vds_start} stop={vds_stop} step={vds_step} {{",
                f"        {_NOISE_DRAIN} noise values=[{values}] oprobe=VDS",
            ]
            if self._gate_noise_requested():
                lines.append(f"        {_NOISE_GATE} noise values=[{values}] oprobe=VGS")
            lines += ["    }", "}"]
        return lines

    def _noise_requested(self):
        table = getattr(self, "parameter_table", {})
        return any(v == _NOISE_MARKER for v in table.values())

    def _gate_noise_requested(self):
        table = getattr(self, "parameter_table", {})
        return any(p.startswith(SIG + "_") and v == _NOISE_MARKER for p, v in table.items())

    def parse_output(self):
        arrs, plots = parse_file(self.output_file_path)
        analysis = SpectreAnalysis()
        for arr, plot in zip(arrs, plots):
            if _is_noise_block(arr, plot):
                analysis.noise.append((plot, arr))
            else:
                analysis.dc.append(arr)
        return analysis

    def extract_parameters(self, analysis, n_vgs, n_vds):
        # A bare list is the DC-only output (kept for callers predating noise support).
        if not isinstance(analysis, SpectreAnalysis):
            analysis = SpectreAnalysis(dc=list(analysis))

        results = {}
        for p, names in self.parameter_table.items():
            if names == _NOISE_MARKER:
                continue
            # a model may name an output differently: the first name present wins
            for col_name in (names,) if isinstance(names, str) else names:
                try:
                    if len(analysis.dc) == 1:
                        data = np.array(analysis.dc[0][col_name]).reshape(n_vgs, n_vds)
                    else:
                        data = np.array([arr[col_name] for arr in analysis.dc])
                    results[p] = data.astype(np.float32)
                    break
                except (KeyError, ValueError):
                    pass

        if self._noise_requested():
            results.update(self.extract_noise(analysis.noise, n_vgs, n_vds))
        return results

    # --- noise extraction -------------------------------------------------------------------
    def extract_noise(self, noise_blocks, n_vgs, n_vds):
        """Turn the noise plot blocks of one job into (n_vgs, n_vds) noise arrays."""
        drain, gate = _split_noise_blocks(noise_blocks, self._gate_noise_requested())
        n_freq = len(self.noise_frequencies)

        sid = self._noise_grid(drain, n_vgs, n_vds, n_freq, "drain (oprobe=VDS)")
        out = {}
        for k, f in enumerate(self.noise_frequencies):
            out[noise_parameter_name(SID, f)] = sid[:, :, k]

        k_th = int(np.argmin([abs(f - self.noise_thermal_frequency) for f in self.noise_frequencies]))
        sid_th = sid[:, :, k_th]
        out[SID_THERMAL] = sid_th
        for k, f in enumerate(self.noise_frequencies):
            if f in self._flicker_frequencies():
                out[noise_parameter_name(SID_FLICKER, f)] = np.maximum(sid[:, :, k] - sid_th, 0.0)

        if self._gate_noise_requested():
            sig = self._noise_grid(gate, n_vgs, n_vds, n_freq, "gate (oprobe=VGS)")
            for k, f in enumerate(self.noise_frequencies):
                out[noise_parameter_name(SIG, f)] = sig[:, :, k]

        # Only keep what the user (or the defaults) asked for.
        return {p: v.astype(np.float32) for p, v in out.items() if p in self.parameter_table}

    def _noise_grid(self, blocks, n_vgs, n_vds, n_freq, what):
        expected = n_vgs * n_vds
        if len(blocks) != expected:
            raise ValueError(
                f"Spectre {what} noise: expected {expected} noise blocks "
                f"({n_vgs} VGS x {n_vds} VDS points), found {len(blocks)}."
            )
        rows = []
        for plot, arr in blocks:
            col = _noise_output_column(arr)
            values = np.asarray(arr[col])
            values = np.abs(values) if np.iscomplexobj(values) else values.astype(float)
            if values.size != n_freq:
                raise ValueError(
                    f"Spectre {what} noise block has {values.size} frequency points, "
                    f"expected {n_freq}."
                )
            _check_frequencies(arr, self.noise_frequencies)
            rows.append(values**2 if self.NOISE_OUTPUT_IS_ROOT_PSD else values)
        return np.asarray(rows, dtype=float).reshape(n_vgs, n_vds, n_freq)


# --- block helpers ---------------------------------------------------------------------------
def _plotname(plot):
    name = plot.get(b"plotname", b"") if isinstance(plot, dict) else b""
    return name.decode("ascii", errors="replace") if isinstance(name, bytes) else str(name)


def _is_noise_block(arr, plot):
    names = arr.dtype.names or ()
    first = names[0].lower() if names else ""
    return first.startswith("freq") or "noise" in _plotname(plot).lower()


def _split_noise_blocks(blocks, with_gate):
    """Separate drain (oprobe=VDS) from gate (oprobe=VGS) noise blocks.

    Uses the analysis name in each block's plotname when Spectre writes it there;
    otherwise relies on execution order (drain, gate, drain, gate, ... per bias point).
    """
    if not with_gate:
        return [b for b in blocks if _NOISE_GATE not in _plotname(b[0])], []
    named_d = [b for b in blocks if _NOISE_DRAIN in _plotname(b[0])]
    named_g = [b for b in blocks if _NOISE_GATE in _plotname(b[0])]
    if named_d and named_g and len(named_d) + len(named_g) == len(blocks):
        return named_d, named_g
    if len(blocks) % 2:
        raise ValueError(
            f"Spectre noise output has {len(blocks)} blocks; expected drain/gate pairs."
        )
    return blocks[0::2], blocks[1::2]


def _noise_output_column(arr):
    names = arr.dtype.names or ()
    for n in names:
        if n.lower() == "out":
            return n
    for n in names:
        if "out" in n.lower():
            return n
    raise ValueError(f"No noise output column ('out') in Spectre noise block; columns: {names}.")


def _check_frequencies(arr, expected):
    names = arr.dtype.names or ()
    if not names or not names[0].lower().startswith("freq"):
        return
    got = np.real(np.asarray(arr[names[0]], dtype=complex)).astype(float)
    if got.size == len(expected) and not np.allclose(np.sort(got), expected, rtol=1e-6):
        warnings.warn(
            f"Spectre noise frequencies {got.tolist()} differ from the requested "
            f"{list(expected)}; values are stored under the requested frequencies.",
            UserWarning,
            stacklevel=4,
        )
