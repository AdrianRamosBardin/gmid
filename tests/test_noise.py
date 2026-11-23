"""Noise data: Spectre netlist/parsing, LUT generation, expressions, and optimizer lookups.

Spectre itself is not available in CI, so its nutbin output is synthesised in the
documented format (ASCII header + big-endian float64 rows, one plot block per analysis).
"""

from __future__ import annotations

import math
import queue
import re
import threading
import warnings

import numpy as np
import pytest

import synthetic
from mosplot.expressions.noise import (
    BOLTZMANN,
    frequency_tag,
    integrate_power_law,
    loglog_interpolator,
    noise_parameter_name,
)
from mosplot.lookup_table_generator import LookupTableGenerator, TransistorSweep
from mosplot.lookup_table_generator.simulators import SpectreSimulator
from mosplot.lookup_table_generator.simulators.spice_simulators import mosfet_simulation
from mosplot.lookup_table_generator.simulators.spice_simulators.spectre_simulator import (
    SpectreAnalysis,
)
from mosplot.lookup_table_generator.simulators.spice_simulators.utils import (
    list_to_string,
    sweep_axis,
)
from mosplot.optimizer import DeviceTable, FastMosfet, NoisePoint
from mosplot.plot import Mosfet, load_lookup_table

FREQS = [1.0, 1e3, 1e6, 1e9]
TEMP = 27.0
KT4 = 4 * BOLTZMANN * (TEMP + 273.15)
GAMMA0 = 0.8
KF = 1e-25  # flicker: S_vg,fl = KF / (W L f)  [V^2/Hz]


# --- naming / math helpers ----------------------------------------------------------------
class TestHelpers:
    @pytest.mark.parametrize(
        "f, tag",
        [(1, "1hz"), (10, "10hz"), (1e3, "1khz"), (2.5e3, "2p5khz"), (1e6, "1meghz"),
         (1e9, "1ghz"), (0.1, "100mhz"), (3e12, "3thz")],
    )
    def test_frequency_tag(self, f, tag):
        assert frequency_tag(f) == tag

    def test_names_are_lowercase(self):
        # The generator lowercases parameter names; noise names must survive that.
        for f in FREQS:
            for kind in ("sid", "sig", "sid_flicker"):
                name = noise_parameter_name(kind, f)
                assert name == name.lower()

    def test_loglog_exact_for_power_law(self):
        fn = loglog_interpolator(1e2, 1e4, 1e3)
        assert fn(1e-20, 1e-22) == pytest.approx(1e-21)  # 1/f
        assert fn(np.array([0.0]), np.array([2.0]))[0] == pytest.approx(1.0)  # linear fallback

    def test_integrate_power_law(self):
        f = np.array([1.0, 1e3, 1e6])
        assert integrate_power_law(f, 1e-18 / f, 1.0, 1e6) == pytest.approx(1e-18 * math.log(1e6))
        assert integrate_power_law(f, np.full(3, 2e-20), 10.0, 1e5) == pytest.approx(2e-20 * (1e5 - 10))
        with pytest.raises(ValueError, match="outside"):
            integrate_power_law(f, 1e-18 / f, 0.1, 1e3)


# --- Spectre netlist ----------------------------------------------------------------------
NMOS = TransistorSweep("nmos", (0.1, 0.5, 0.1), (0.1, 0.3, 0.1), (0, -0.2, -0.2), [1e-6, 2e-6])


def _spectre(**kw):
    return SpectreSimulator(simulator_path="true", noise_frequencies=FREQS, **kw)


class TestSpectreNoiseSetup:
    def test_default_has_no_noise(self):
        sim = SpectreSimulator(simulator_path="true")
        assert sim.noise_parameter_names() == []
        assert "noise" not in list_to_string(sim.setup_dc_simulation(NMOS))
        assert sim.noise_metadata() is None

    def test_netlist_has_nested_noise_sweep(self):
        sim = _spectre()
        with warnings.catch_warnings():
            warnings.simplefilter("error")  # noise params must not trigger "unsupported"
            text = list_to_string(sim.setup_dc_simulation(NMOS))
        assert "nswp_vgs sweep param=dc dev=VGS start=0.1 stop=0.5 step=0.1 {" in text
        assert "nswp_vds sweep param=dc dev=VDS start=0.1 stop=0.3 step=0.1 {" in text
        assert "noise_id noise values=[1 1000 1000000 1000000000] oprobe=VDS" in text
        assert "noise_ig noise values=[1 1000 1000000 1000000000] oprobe=VGS" in text

    def test_gate_noise_off(self):
        sim = _spectre(gate_noise=False)
        text = list_to_string(sim.setup_dc_simulation(NMOS))
        assert "noise_ig" not in text
        assert not any(p.startswith("sig_") for p in sim.parameters_to_save)

    def test_parameter_names(self):
        sim = _spectre()
        names = sim.noise_parameter_names()
        assert names[:4] == ["sid_1hz", "sid_1khz", "sid_1meghz", "sid_1ghz"]
        assert "sid_thermal" in names
        # flicker only below the thermal reference frequency
        assert [n for n in names if n.startswith("sid_flicker")] == [
            "sid_flicker_1hz", "sid_flicker_1khz", "sid_flicker_1meghz"
        ]
        assert all(n in sim.parameters_to_save for n in names)

    def test_thermal_frequency_added_if_missing(self):
        sim = SpectreSimulator(simulator_path="true", noise_frequencies=[10, 1e3], noise_thermal_frequency=1e8)
        assert sim.noise_frequencies == [10.0, 1e3, 1e8]
        assert sim.noise_thermal_frequency == 1e8

    def test_clone_keeps_noise_without_duplicates(self):
        sim = _spectre(gate_noise=False)
        clone = sim.clone()
        assert clone.noise_frequencies == sim.noise_frequencies
        assert clone.gate_noise is False
        assert clone.parameters_to_save == sim.parameters_to_save
        assert len(set(clone.parameters_to_save)) == len(clone.parameters_to_save)

    def test_user_parameters_list_not_mutated(self):
        user = ["id", "gm"]
        _spectre(parameters_to_save=user)
        assert user == ["id", "gm"]

    def test_bad_frequencies(self):
        with pytest.raises(ValueError):
            SpectreSimulator(simulator_path="true", noise_frequencies=[0, 1e3])
        with pytest.raises(ValueError):
            SpectreSimulator(simulator_path="true", noise_thermal_frequency=1e9)


# --- synthetic Spectre nutbin output ---------------------------------------------------------
def write_nutbin(path, blocks):
    """blocks: list of (plotname, [(name, unit, values), ...])."""
    out = b""
    for plotname, cols in blocks:
        n = len(cols[0][2])
        header = [
            "Title: synthetic", "Date: today", f"Plotname: {plotname}", "Flags: real",
            f"No. Variables: {len(cols)}", f"No. Points: {n}",
        ]
        specs = [f"\t{i}\t{name}\t{unit}" for i, (name, unit, _) in enumerate(cols)]
        text = "\n".join(header) + "\nVariables:" + specs[0] + "\n" + "\n".join(specs[1:]) + "\nBinary:\n"
        data = np.column_stack([np.asarray(v, dtype=float) for _, _, v in cols]).astype(">f8")
        out += text.encode() + data.tobytes()
    path.write_bytes(out)


def fake_device(length, vbs, vgs, vds):
    """Analytic per-point values for the fake Spectre run."""
    gm = 1e-3 * vgs * (1 + 0.1 * vds) * (1e-6 / length) * (1 - 0.1 * vbs)
    id_ = 0.1 * gm * vgs
    sth = KT4 * GAMMA0 * gm
    k = gm**2 * KF / (10e-6 * length)  # S_id,fl = K / f
    sig = 1e-32 * (1 + vgs)
    return gm, id_, sth, k, sig


def fake_spectre_blocks(length, vbs, sweep, with_names=True, with_gate=True, sig_zero=False):
    vgs_axis, vds_axis = sweep_axis(sweep.vgs), sweep_axis(sweep.vds)
    blocks = []
    for vgs in vgs_axis:
        gm, id_, *_ = fake_device(length, vbs, vgs, vds_axis)
        blocks.append((
            "vgs_swp-000_vds_dc: DC Analysis `vds_dc'",
            [("dc", "V", vds_axis), ("VDS:n", "A", -id_), ("M1:gm", "S", gm)],
        ))
    f = np.array(FREQS)
    for vgs in vgs_axis:
        for vds in vds_axis:
            _, _, sth, k, sig = fake_device(length, vbs, vgs, vds)
            sid = sth + k / f
            d_name = "nswp_vgs-000_nswp_vds-000_noise_id: Noise Analysis" if with_names else "Noise Analysis"
            g_name = "nswp_vgs-000_nswp_vds-000_noise_ig: Noise Analysis" if with_names else "Noise Analysis"
            blocks.append((d_name, [("freq", "Hz", f), ("out", "A/sqrt(Hz)", np.sqrt(sid))]))
            if with_gate:
                g = np.zeros_like(f) if sig_zero else np.full_like(f, sig)
                blocks.append((g_name, [("freq", "Hz", f), ("out", "A/sqrt(Hz)", np.sqrt(g))]))
    return blocks


class TestSpectreNoiseParsing:
    @pytest.mark.parametrize("with_names", [True, False])
    def test_parse_and_extract(self, tmp_path, with_names):
        sim = _spectre()
        sim.setup_dc_simulation(NMOS)
        sim.output_file_path = str(tmp_path / "out.raw")
        write_nutbin(tmp_path / "out.raw", fake_spectre_blocks(1e-6, 0.0, NMOS, with_names=with_names))

        analysis = sim.parse_output()
        assert isinstance(analysis, SpectreAnalysis)
        n_vgs, n_vds = len(sweep_axis(NMOS.vgs)), len(sweep_axis(NMOS.vds))
        assert len(analysis.dc) == n_vgs and len(analysis.noise) == 2 * n_vgs * n_vds

        res = sim.extract_parameters(analysis, n_vgs, n_vds)
        vgs, vds = np.meshgrid(sweep_axis(NMOS.vgs), sweep_axis(NMOS.vds), indexing="ij")
        gm, id_, sth, k, sig = fake_device(1e-6, 0.0, vgs, vds)
        assert res["gm"].shape == (n_vgs, n_vds)
        np.testing.assert_allclose(res["gm"], gm, rtol=1e-6)
        np.testing.assert_allclose(res["sid_1khz"], sth + k / 1e3, rtol=1e-6)
        np.testing.assert_allclose(res["sid_thermal"], sth + k / 1e9, rtol=1e-6)
        np.testing.assert_allclose(res["sid_flicker_1hz"], k * (1 - 1e-9), rtol=1e-5)
        np.testing.assert_allclose(res["sig_1meghz"], sig, rtol=1e-6)
        assert "sid_flicker_1ghz" not in res
        assert all(v.dtype == np.float32 for v in res.values())

    def test_dc_only_list_still_accepted(self):
        sim = SpectreSimulator(simulator_path="true", parameters_to_save=["gm"])
        sim.setup_dc_simulation(NMOS)
        res = sim.extract_parameters([{"M1:gm": np.arange(15.0)}], n_vgs=5, n_vds=3)
        assert res["gm"].shape == (5, 3)

    def test_wrong_block_count_raises(self, tmp_path):
        sim = _spectre()
        sim.setup_dc_simulation(NMOS)
        sim.output_file_path = str(tmp_path / "out.raw")
        blocks = fake_spectre_blocks(1e-6, 0.0, NMOS)[:-2]  # drop one bias point
        write_nutbin(tmp_path / "out.raw", blocks)
        with pytest.raises(ValueError, match="noise blocks"):
            sim.extract_parameters(sim.parse_output(), 5, 3)

    def test_psd_output_switch(self, tmp_path, monkeypatch):
        sim = _spectre(gate_noise=False)
        monkeypatch.setattr(SpectreSimulator, "NOISE_OUTPUT_IS_ROOT_PSD", False)
        sim.setup_dc_simulation(NMOS)
        sim.output_file_path = str(tmp_path / "out.raw")
        write_nutbin(tmp_path / "out.raw", fake_spectre_blocks(1e-6, 0.0, NMOS, with_gate=False))
        res = sim.extract_parameters(sim.parse_output(), 5, 3)
        _, _, sth, k, _ = fake_device(1e-6, 0.0, 0.1, 0.1)
        assert res["sid_1khz"][0, 0] == pytest.approx(math.sqrt(sth + k / 1e3), rel=1e-5)


# --- full generator run with a fake Spectre ---------------------------------------------------
class FakeSpectre(SpectreSimulator):
    """Writes synthetic nutbin output instead of running Spectre."""

    SIG_ZERO = False

    def run_simulation(self, netlist, verbose=False):
        text = list_to_string(netlist)
        length = float(re.search(r" l=(\S+)", text).group(1))
        vbs = float(re.search(r"VBS \(nb 0\) vsource dc=(\S+)", text).group(1))
        from pathlib import Path

        write_nutbin(Path(self.output_file_path),
                     fake_spectre_blocks(length, vbs, NMOS, sig_zero=self.SIG_ZERO))


@pytest.fixture
def inline_workers(monkeypatch):
    # Run the generator's worker processes as threads so the fake simulator class
    # needs no pickling and the test stays fast.
    monkeypatch.setattr(mosfet_simulation, "Process", threading.Thread)
    monkeypatch.setattr(mosfet_simulation, "Queue", queue.Queue)


def _build(tmp_path, sig_zero=False):
    FakeSpectre.SIG_ZERO = sig_zero
    try:
        sim = FakeSpectre(simulator_path="true", noise_frequencies=FREQS, temperature=TEMP,
                          parameters_to_save=["id", "gm"])
        gen = LookupTableGenerator(simulator=sim, model_sweeps={"nch": NMOS}, n_process=2)
        gen.build(str(tmp_path / "lut"))
    finally:
        FakeSpectre.SIG_ZERO = False
    return load_lookup_table(str(tmp_path / "lut.npz"))


class TestGeneratorWithNoise:
    def test_build_stores_noise_like_any_parameter(self, tmp_path, inline_workers, capsys):
        table = _build(tmp_path)
        dev = table["nch"]
        assert dev["noise"]["frequencies"] == FREQS
        assert dev["noise"]["thermal_frequency"] == 1e9
        assert dev["noise"]["temperature"] == TEMP
        for p in ("sid_1hz", "sid_thermal", "sid_flicker_1khz", "sig_1ghz"):
            assert p in dev["parameter_names"]
            assert dev[p].shape == (2, 2, 5, 3)  # length x vbs x vgs x vds

        # L = 2u, vbs = -0.2, vgs = 0.3, vds = 0.2
        _, _, sth, k, _ = fake_device(2e-6, -0.2, 0.3, 0.2)
        assert dev["sid_1khz"][1, 1, 2, 1] == pytest.approx(sth + k / 1e3, rel=1e-5)

    def test_all_zero_gate_noise_is_dropped(self, tmp_path, inline_workers, capsys):
        table = _build(tmp_path, sig_zero=True)
        dev = table["nch"]
        assert not any(p.startswith("sig_") for p in dev["parameter_names"])
        nmos = Mosfet(lookup_table=table, mos="nch", vbs=0.0, vds=0.2, vgs=(0.1, 0.5))
        with pytest.raises(ValueError, match="no 'sig'"):
            nmos.sig_expression(1e3)
        assert nmos.sid_expression(1e3).variables == ["sid_1khz"]

    def test_mosfet_noise_expressions(self, tmp_path, inline_workers, capsys):
        table = _build(tmp_path)
        nmos = Mosfet(lookup_table=table, mos="nch", vbs=0.0, vds=0.2, vgs=(0.1, 0.5))
        assert nmos.has_noise and nmos.noise_frequencies == FREQS
        vgs = sweep_axis(NMOS.vgs)
        gm, _, sth, k, sig = fake_device(np.array([1e-6, 2e-6])[:, None], 0.0, vgs[None, :], 0.2)

        sid, _ = nmos.calculate_from_expression(nmos.sid_expression(1e3))
        np.testing.assert_allclose(sid, sth + k / 1e3, rtol=1e-5)
        # between stored points: log-log interpolation (flicker part is ~exactly 1/f)
        fl, _ = nmos.calculate_from_expression(nmos.sid_flicker_expression(3e4))
        np.testing.assert_allclose(fl, k / 3e4, rtol=1e-3)
        svg, _ = nmos.calculate_from_expression(nmos.svg_expression(1e3))
        np.testing.assert_allclose(svg, (sth + k / 1e3) / gm**2, rtol=1e-5)
        gamma, _ = nmos.calculate_from_expression(nmos.gamma_expression)
        np.testing.assert_allclose(gamma, GAMMA0, rtol=1e-4)
        fc, _ = nmos.calculate_from_expression(nmos.flicker_corner_expression)
        np.testing.assert_allclose(fc, k / sth, rtol=1e-3)
        g, _ = nmos.calculate_from_expression(nmos.sig_expression(1e6))
        np.testing.assert_allclose(g, np.broadcast_to(sig, g.shape), rtol=1e-5)

        # raw names work in custom expressions and in the plotting helpers
        assert nmos.noise_parameter("sid", 1e6) == "sid_1meghz"
        res = nmos.lookup_expression_from_table(
            length=1e-6, vbs=0.0, vds=0.2, vgs=(0.1, 0.5, 0.1), primary="vgs",
            expression=nmos.svg_thermal_expression,
        )
        assert res.shape == (5,)
        with pytest.raises(ValueError, match="outside"):
            nmos.sid_expression(1e10)


# --- analytic noise on the square-law LUT: FastMosfet / DeviceTable -------------------------
def add_noise(dev, freqs=FREQS):
    """Add noise columns the way the Spectre backend computes them."""
    dev = dict(dev)
    gm = np.abs(dev["gm"])
    L = dev["length"][:, None, None, None]
    sth_true = KT4 * GAMMA0 * gm
    k = gm**2 * KF / (synthetic.WIDTH * L)
    sid = {f: sth_true + k / f for f in freqs}
    sth = sid[freqs[-1]]
    names = list(dev["parameter_names"])
    for f in freqs:
        dev[noise_parameter_name("sid", f)] = sid[f]
        dev[noise_parameter_name("sig", f)] = 1e-30 * (f / 1e3) ** 2 * np.ones_like(gm)
        names += [noise_parameter_name("sid", f), noise_parameter_name("sig", f)]
        if f < freqs[-1]:
            dev[noise_parameter_name("sid_flicker", f)] = np.maximum(sid[f] - sth, 0)
            names.append(noise_parameter_name("sid_flicker", f))
    dev["sid_thermal"] = sth
    names.append("sid_thermal")
    dev["parameter_names"] = names
    dev["noise"] = {"frequencies": list(freqs), "thermal_frequency": freqs[-1],
                    "temperature": TEMP, "gate_noise": True}
    return dev


@pytest.fixture(scope="module")
def noisy_table(lookup_table):
    t = dict(lookup_table)
    t["nch"] = add_noise(lookup_table["nch"])
    t["pch"] = add_noise(lookup_table["pch"])
    return t


class TestOptimizerNoise:
    def test_fast_mosfet_expressions(self, noisy_table):
        fm = FastMosfet(noisy_table, "nch")
        assert fm.has_noise
        gamma = fm.interpolate(400e-9, 12.0, 0.6, 0.0, fm.gamma_expression)
        assert gamma == pytest.approx(GAMMA0, rel=1e-3)
        truth = synthetic.analytic_from_gmid(400e-9, 12.0, 0.6, 0.0)
        svg = fm.interpolate(400e-9, 12.0, 0.6, 0.0, fm.svg_thermal_expression)
        assert svg == pytest.approx(KT4 * GAMMA0 / truth["gm"], rel=2e-2)

    def test_device_table_noise(self, noisy_table):
        fm = FastMosfet(noisy_table, "nch")
        dt = DeviceTable(fm, "n")
        assert dt.has_noise
        npt = dt.noise(gmid=12.0, L=400e-9, vds=0.6)
        assert isinstance(npt, NoisePoint)
        truth = synthetic.analytic_from_gmid(400e-9, 12.0, 0.6, 0.0)
        gm = truth["gm"]
        k = gm**2 * KF / (synthetic.WIDTH * 400e-9)
        assert npt.sid_thermal == pytest.approx(KT4 * GAMMA0 * gm, rel=2e-2)
        assert npt.sid_at(1e3) == pytest.approx(KT4 * GAMMA0 * gm + k / 1e3, rel=3e-2)
        assert npt.sig_at(1e6) == pytest.approx(1e-30 * 1e6, rel=1e-6)
        assert npt.flicker_corner() == pytest.approx(k / (KT4 * GAMMA0 * gm), rel=3e-2)

        # gamma uses the same-width gm; scaling doubles every PSD, leaves fc unchanged
        op = dt(gmid=12.0, L=400e-9, vds=0.6)
        gm_table = 12.0 * op.jd * dt.table_width
        # gmid * |id| vs the interpolated gm: equal up to grid-interpolation error
        assert npt.gamma(gm_table) == pytest.approx(GAMMA0, rel=1e-2)
        big = npt.scaled(2 * dt.table_width, dt.table_width)
        assert big.sid_at(1e3) == pytest.approx(2 * npt.sid_at(1e3))
        assert big.sig_at(1e3) == pytest.approx(2 * npt.sig_at(1e3))
        assert big.flicker_corner() == pytest.approx(npt.flicker_corner())
        assert big.gamma(2 * gm_table) == pytest.approx(npt.gamma(gm_table))
        assert npt.svg_at(1e3, gm_table) == pytest.approx(npt.sid_at(1e3) / gm_table**2)

        # integrated noise: white band is exact; result is positive and finite
        band = npt.integrated_sid(1e6, 1e9)
        assert band == pytest.approx(npt.sid_thermal * (1e9 - 1e6), rel=1e-2)
        assert npt.integrated_svg(1.0, 1e6, gm_table) > 0

    def test_pmos_device_table_noise(self, noisy_table):
        dt = DeviceTable(FastMosfet(noisy_table, "pch"), "p")
        npt = dt.noise(gmid=12.0, L=400e-9, vds=0.6)
        assert np.isfinite(npt.sid_thermal) and npt.sid_thermal > 0

    def test_no_noise_table(self, lookup_table):
        fm = FastMosfet(lookup_table, "nch")
        dt = DeviceTable(fm, "n")
        assert not dt.has_noise and not fm.has_noise
        with pytest.raises(ValueError, match="no noise data"):
            dt.noise(gmid=12.0, L=400e-9, vds=0.6)
        with pytest.raises(ValueError, match="no noise data"):
            fm.sid_expression(1e3)

    def test_parallel_worker_state_is_picklable(self, noisy_table):
        import pickle

        from mosplot.expressions import Expression

        fm = FastMosfet(noisy_table, "nch")
        dt = DeviceTable(fm, "n")
        # mirrors what optimizer/parallel.py ships to each worker
        pickle.dumps({k: v for k, v in vars(dt).items() if k not in ("fm", "_lut", "_needed_data")})
        pickle.dumps({k: v for k, v in vars(fm).items()
                      if k != "_table" and not isinstance(v, Expression)})
