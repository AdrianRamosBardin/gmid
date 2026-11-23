# Generating a Lookup Table

Everything in Mosplot runs on a lookup table: a pre-computed record of how a
transistor behaves at every bias point. You build one per technology and corner,
save it to a `.npz`, and reuse it for plotting and optimization.

For each model, the generator sweeps a single device and stores every requested
operating-point parameter (`id`, `gm`, `gds`, `cgg`, ...) as a 4-D array indexed
by

```
length × vbs × vgs × vds
```

The device width is fixed (e.g. 10 µm); quantities that scale with width are
normalized by it later.

## 1. Configure a simulator

ngspice, HSPICE, and Spectre take the same arguments; only the class name
changes.

```python
from mosplot.lookup_table_generator.simulators import NgspiceSimulator

sim = NgspiceSimulator(
    simulator_path="ngspice",          # binary name or full path
    temperature=27,                    # °C
    include_paths=[                    # model files to .include ...
        "./NMOS_VTH.inc",
        "./PMOS_VTH.inc",
    ],
    device_parameters={"w": 10e-6},    # fixed width (and any other instance params)
)
```

```python
from mosplot.lookup_table_generator.simulators import HspiceSimulator

sim = HspiceSimulator(
    simulator_path="hspice",
    lib_mappings=[("/path/to/models.lib", "tt")],
    # Transistor wrapped in a subcircuit: (instance name, hierarchical device name)
    mos_spice_symbols=("x1", "x1.main"),
    device_parameters={"w": 10e-6},
)
```

```python
from mosplot.lookup_table_generator.simulators import SpectreSimulator

sim = SpectreSimulator(
    simulator_path="spectre",
    lib_mappings=[("/path/to/models.scs", "tt")],
    device_parameters={"w": 10e-6},
)
```

Arguments:

| Argument             | Default                                | Meaning                                                                              |
| ---                  | ---                                    | ---                                                                                  |
| `simulator_path`     | `"ngspice"` / `"hspice"` / `"spectre"` | Simulator binary. Must be on `PATH` or a full path.                                  |
| `temperature`        | `27`                                   | Simulation temperature in °C.                                                        |
| `include_paths`      | `None`                                 | Model files to include.                                                              |
| `lib_mappings`       | `None`                                 | List of `(library file, section)` pairs, e.g. for process corners.                   |
| `device_parameters`  | `{"w": 10e-6}`                         | Instance parameters written on the transistor line (`w`, `nf`, ...).                 |
| `parameters_to_save` | see below                              | Operating-point parameters to save.                                                  |
| `mos_spice_symbols`  | `("m1", "m1")`                         | Instance name, and the name used to probe it. Change when the model is a subcircuit. |
| `osdi_paths`         | `None`                                 | ngspice only. Compiled Verilog-A models (`.osdi`) to load.                           |
| `hdl_paths`          | `None`                                 | HSPICE only. Verilog-A files to load.                                                |
| `raw_spice`          | `None`                                 | Extra netlist lines, pasted verbatim.                                                |
| `noise_frequencies`  | `None`                                 | Spectre only. Frequencies (Hz) at which to store noise; see [Noise](#noise-spectre). |
| `noise_thermal_frequency` | highest of `noise_frequencies`    | Spectre only. Frequency whose drain noise is taken as the thermal (white) level.     |
| `gate_noise`         | `True`                                 | Spectre only. Also store gate-current noise.                                         |

At least one of `include_paths` or `lib_mappings` is required. Paths are checked
before anything runs.

Available parameters:

| Simulator | Available |
|---|---|
| ngspice | `id`, `weff`, `vth`, `vdsat`, `vdssat`, `gm`, `gmbs`, `gds`, `cgg`, `cgs`, `cbg`, `cgd`, `cdd` |
| HSPICE | `id`, `vth`, `vdsat`, `gm`, `gmbs`, `gds`, `cgg`, `cgs`, `cgd`, `cgb`, `cdd`, `css` |
| Spectre | `id`, `vth`, `vdsat`, `gm`, `gmbs`, `gds`, `cgg`, `cgs`, `cgd`, `cgb`, `cdd`, `css` |

By default everything except `css` is saved. If your model does not expose a
parameter, pass the ones you want in `parameters_to_save` rather than editing
the repo.

### Noise (Spectre)

Noise depends on frequency, so it is stored at a fixed list of frequencies,
with each frequency as its own 4-D parameter. These parameters behave like `gm`
or `cgg` everywhere: plots, custom expressions, interpolation and the
optimizer.

```python
sim = SpectreSimulator(
    lib_mappings=[("/path/to/models.scs", "tt")],
    noise_frequencies=[1, 1e3, 1e6, 1e9],   # Hz
    # noise_thermal_frequency=1e9,          # default: the highest frequency
    # gate_noise=True,
)
```

Inside the DC sweep, a Spectre `noise` analysis runs at every (VGS, VDS) point.
Gate and drain are held by ideal sources, so the noise current through them is
the device's short-circuit noise:

| Parameter | Quantity | Unit |
|---|---|---|
| `sid_<f>` | total drain-current noise (`oprobe=VDS`) | A²/Hz |
| `sig_<f>` | gate-current noise (`oprobe=VGS`): induced gate and gate-leakage shot noise, if the model has them | A²/Hz |
| `sid_thermal` | `sid` at `noise_thermal_frequency`, i.e. the white level | A²/Hz |
| `sid_flicker_<f>` | `max(sid_<f> − sid_thermal, 0)` for every lower frequency | A²/Hz |

`<f>` is a lowercase tag: `1hz`, `1khz`, `1meghz`, `1ghz`, `2p5khz` (2.5 kHz),
`100mhz` (100 mHz). Example: `sid_1khz`.

The thermal/flicker split is derived from the frequency points, so it works
with any model (BSIM, PSP, BSIM-CMG). Two conditions apply:

- **Thermal frequency above the flicker corner.** It must be above the flicker
  corner at every bias point you sweep. Long devices at low gm/ID have the
  highest corners.
- **Flat high-frequency spectrum.** The drain noise must still be flat there.
  1 GHz is usually fine; check the `sid_<f>` curves if your devices are very
  long.

A noise analysis at every bias point is much slower than the DC sweep, because
it adds two small-signal analyses per (VGS, VDS) point. Use few frequencies and
a coarser VGS/VDS step if build time matters. Noise is not available with
ngspice or HSPICE.

Each device entry of the table also gets a `"noise"` dict with the frequencies,
the thermal frequency and the simulation temperature. The noise expressions use
it (see [design charts](plotting.md#noise)).

> Spectre writes the noise output as a root spectral density (A/√Hz), which is
> squared into A²/Hz. If your Spectre version already writes A²/Hz, set
> `SpectreSimulator.NOISE_OUTPUT_IS_ROOT_PSD = False` before building.

## 2. Define the sweeps

One `TransistorSweep` per model. Voltages are `(start, stop, step)` and follow
the device's own sign convention, so PMOS sweeps go negative:

```python
from mosplot.lookup_table_generator import TransistorSweep

nmos_sweep = TransistorSweep(
    mos_type="nmos",
    vgs=(0, 1.2, 0.01),
    vds=(0.01, 1.2, 0.01),
    vbs=(0, -1.2, -0.1),
    length=[130e-9, 180e-9, 250e-9, 500e-9, 1e-6, 2e-6, 5e-6],
)

pmos_sweep = TransistorSweep(
    mos_type="pmos",
    vgs=(0, -1.2, -0.01),
    vds=(-0.01, -1.2, -0.01),
    vbs=(0, 1.2, 0.1),
    length=[130e-9, 180e-9, 250e-9, 500e-9, 1e-6, 2e-6, 5e-6],
)
```

## 3. Build the table

```python
from mosplot.lookup_table_generator import LookupTableGenerator

gen = LookupTableGenerator(
    description="my process, tt, 27C",
    simulator=sim,
    model_sweeps={
        "NMOS_VTH": nmos_sweep,   # key = model name, as it appears in the model file
        "PMOS_VTH": pmos_sweep,
    },
    n_process=1,
)

gen.op_simulation()           # optional: run one operating point and print the netlist
gen.build("./my_process_tt")  # writes ./my_process_tt.npz
```

The keys of `model_sweeps` are the model names written into the netlist and the
names used later to pick a device (`Mosfet(..., mos="NMOS_VTH")`).
`op_simulation()` is a quick sanity check when setting up a new PDK.
`n_process` runs several simulations in parallel; keep it at `1` for ngspice,
which already uses multiple threads internally.

For process corners, build one table per corner, changing only `lib_mappings`
and the file name:

```python
sweeps = {"NMOS_VTH": nmos_sweep, "PMOS_VTH": pmos_sweep}

for corner in ["tt", "ss", "ff"]:
    sim = SpectreSimulator(lib_mappings=[("/path/to/models.scs", corner)])
    gen = LookupTableGenerator(description=corner, simulator=sim, model_sweeps=sweeps)
    gen.build(f"./luts/my_process_{corner}")
```

The optimizer uses these tables to size across all corners at once.

## Loading a table

```python
from mosplot.plot import load_lookup_table

lookup_table = load_lookup_table("./my_process_tt.npz")
print(lookup_table.keys())
# dict_keys(['NMOS_VTH', 'PMOS_VTH', 'description', 'simulator', 'parameter_names', 'device_parameters'])
```

The file is a plain Python dict. Next step: [design charts](plotting.md). A
complete runnable script is in
[`examples/lookup_table_generator`](../examples/lookup_table_generator).
