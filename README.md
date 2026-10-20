# bayesflow-irt

[![GitHub release](https://img.shields.io/github/v/release/matthiaskloft/bayesflow-irt)](https://github.com/matthiaskloft/bayesflow-irt/releases/latest)
[![PyPI version](https://img.shields.io/pypi/v/bayesflow-irt)](https://pypi.org/project/bayesflow-irt/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-blue.svg)](https://www.python.org/downloads/)

> [!NOTE]
> All 0.x.x releases are under active development and may introduce breaking
> API changes.

A [BayesFlow](https://github.com/bayesflow-org/bayesflow) add-on for amortized
Bayesian inference in Item Response Theory (IRT) models: train a network once on
simulated response matrices, then draw posteriors for new data sets without
refitting, for any number of items and persons within the trained range.

## What is included

- Variable-size simulators and adapters for the 1PL, 2PL, PCM, and GPCM models
- `MatrixEquivariantIRTSummary`, a summary network that is exchangeable over
  both items and persons (Hartford-style matrix blocks)
- `EquivariantIRTApproximator` for per-item posterior inference
- Person scoring, metadata sidecars for trained models, and memory controls for
  large `n_persons`
- Optional: calibration-aware training (`CalibratedIRTApproximator`) and
  hyperparameter-optimization hooks (`bayesflow_irt.hpo`)
- Planned: pretrained checkpoints with model cards, and example notebooks

## Installation

The package is not on PyPI yet. Install it from a clone:

```bash
git clone https://github.com/matthiaskloft/bayesflow-irt.git
cd bayesflow-irt
pip install -e ".[torch]"
```

Extras: `dev` (tests), `examples` (plotting and Jupyter), `card` (model-card
plots), `hpo` (`bayesflow-hpo`, from PyPI), `numba` (faster response
sampling), and `calibration` (`bayesflow-calibration-loss`, which will be
released separately; until then this extra cannot be installed). The
`tensorflow` and `jax` backend extras are also available.

Set the Keras backend before importing `keras`, `bayesflow`, or `bayesflow_irt`:

```bash
export KERAS_BACKEND=torch          # PowerShell: $env:KERAS_BACKEND = "torch"
```

## Quick start

```python
import bayesflow as bf
from bayesflow_irt import (
    EquivariantIRTApproximator,
    MatrixEquivariantIRTSummary,
    Prior,
    make_2pl_adapter,
    make_2pl_simulator,
)

simulator = make_2pl_simulator(
    n_items=(10, 25),
    n_persons=(100, 250),
    prior_a=Prior("lognormal", loc=0.0, scale=0.3),
    prior_b=Prior("normal", loc=0.0, scale=2.0),
)
adapter = make_2pl_adapter(simulator, internal_parameterization="loading")

approximator = EquivariantIRTApproximator(
    adapter=adapter,
    inference_network=bf.networks.FlowMatching(),
    summary_network=MatrixEquivariantIRTSummary(embed_dim=64),
)
approximator.compile(optimizer="adam")
approximator.fit(simulator=simulator, epochs=10, num_batches=32, batch_size=64)

batch = simulator.sample((4,))
posterior = approximator.sample(num_samples=200, conditions=batch)
```

The public functions and classes are documented in their docstrings.

## Examples

[examples/train_1pl_minimal.py](examples/train_1pl_minimal.py) is a minimal
end-to-end script (simulate, adapt, train, sample, save). Pretrained
checkpoints are not distributed yet; example notebooks will follow with them.

For longer runs, [scripts/train_irt.py](scripts/train_irt.py) runs the same
training pipeline from a JSON config and writes only files:

```bash
python scripts/train_irt.py --config scripts/configs/smoke_1pl.json
```

`scripts/configs/` has a quick smoke test (`smoke_1pl.json`) and a fixed-size
2PL run (`2pl_fixed.json`).

On Windows, run Jupyter with a kernel that imports `torch` first
([scripts/ipykernel_torch_preload.py](scripts/ipykernel_torch_preload.py));
otherwise loading `pyzmq` before `torch` can crash with `WinError 1114`.

## Development

```bash
pip install -e ".[dev,torch]"
pytest tests/
ruff check src/
```

See [CONTRIBUTING.md](CONTRIBUTING.md) and [CHANGELOG.md](CHANGELOG.md).

## License and citation

MIT, see [LICENSE](LICENSE). To cite the software, see
[CITATION.cff](CITATION.cff). The methods the package implements are cited in
[references.md](references.md).
