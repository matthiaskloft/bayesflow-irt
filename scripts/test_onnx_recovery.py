"""Compare native and ONNX 1PL/2PL recovery/calibration with paired posterior draws.

Run after installing the export dependencies (``onnx``, ``onnxruntime``,
``onnxscript``) next to the project::

    python scripts/test_onnx_recovery.py --model path/to/model.keras

Exports both summary and flow velocity networks. Uses the checkpoint's existing
BayesFlow ODE integrator (including its configured adaptive solver) for both
paths, with identical initial base samples, adapters and validation datasets.
This tests ONNX neural inference within the posterior pipeline, not a standalone
ONNX sampler or R implementation. No retraining or solver substitution occurs.

References: BayesFlow's installed FlowMatching._inverse and utils.integrate;
bayesflow_irt.sbc.compute_calibration_error and _compute_recovery_metrics;
https://docs.pytorch.org/docs/stable/onnx.html
https://onnxruntime.ai/docs/api/python/api_summary.html
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path


def main() -> None:
    """Export a checkpoint and assert paired posterior and metric agreement."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model",
        type=Path,
        required=True,
        help="Saved 1PL or 2PL model (.keras) with its metadata sidecar.",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=Path(".tmp/onnx-1pl-recovery")
    )
    parser.add_argument("--datasets", type=int, default=32, help="Datasets per corner")
    parser.add_argument("--draws", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--seed", type=int, default=73)
    parser.add_argument("--metric-atol", type=float, default=1e-4)
    parser.add_argument("--calibration-atol", type=float, default=1e-3)
    args = parser.parse_args()
    if min(args.datasets, args.draws, args.batch_size) < 2:
        parser.error("datasets, draws and batch-size must all be at least 2")
    if min(args.metric_atol, args.calibration_atol) < 0:
        parser.error("Metric tolerances must be non-negative")

    os.environ["KERAS_BACKEND"] = "torch"
    import bayesflow
    import keras
    import numpy as np
    import onnx
    import onnxruntime as ort
    import torch
    from bayesflow.utils import integrate

    import bayesflow_irt as irt
    from bayesflow_irt.sbc import (
        _compute_log_recovery_metrics,
        _compute_recovery_metrics,
    )

    keras.utils.set_random_seed(args.seed)
    torch.set_num_threads(1)
    metadata_path = args.model.with_name(args.model.stem + "_metadata.json")
    # The loader upgrades v1 sidecars, so old and current checkpoints both
    # resolve to one normalized `model_spec` (model, priors, size ranges).
    model_spec = irt.MetadataTracker.load(metadata_path).to_dict()["model_spec"]
    model_family = model_spec["model"].upper()
    if model_family not in {"1PL", "2PL"}:
        parser.error("This validation script supports 1PL and 2PL checkpoints only")
    parameter_names = ("b",) if model_family == "1PL" else ("a", "b")
    param_dim = len(parameter_names)
    loaded = keras.saving.load_model(
        args.model,
        compile=False,
        custom_objects={
            name: getattr(irt, name)
            for name in (
                "EquivariantIRTApproximator",
                "EquivariantIRTSummary",
                "MatrixEquivariantIRTSummary",
            )
            if hasattr(irt, name)
        },
    )
    if loaded.standardize_params:
        parser.error("Online parameter standardization is not supported by this test")
    flow = loaded.inference_network
    flow.eval()
    loaded.summary_network.eval()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    # A report from an earlier run would otherwise still read "passed" next to
    # the ONNX files this run is about to overwrite, until the first corner
    # finishes; an interrupted rerun must leave no passing report behind.
    (args.output_dir / "recovery_report.json").unlink(missing_ok=True)
    subprocess.run(
        [
            sys.executable,
            str(Path(__file__).with_name("test_onnx_export.py")),
            "--model",
            str(args.model),
            "--output-dir",
            str(args.output_dir),
        ],
        check=True,
    )

    class VelocityModule(torch.nn.Module):
        """Export the trained velocity, including its output projector."""

        def __init__(self, network):
            super().__init__()
            self.network = network

        def forward(self, state, time, conditions):
            return self.network.velocity(
                state, time=time, conditions=conditions, training=False
            )

    velocity = VelocityModule(flow).eval()
    condition_dim = (
        loaded.summary_network.compute_output_shape((2, 50, 5))[-1]
        + loaded._global_condition_dim
    )
    row = torch.export.Dim("rows", min=1)
    velocity_path = args.output_dir / "velocity.onnx"
    torch.onnx.export(
        velocity,
        (
            torch.zeros((4, param_dim)),
            torch.zeros((4, 1)),
            torch.zeros((4, condition_dim)),
        ),
        str(velocity_path),
        dynamo=True,
        verbose=False,
        external_data=False,
        input_names=["state", "time", "conditions"],
        output_names=["velocity"],
        dynamic_shapes={name: {0: row} for name in ("state", "time", "conditions")},
    )
    onnx.checker.check_model(str(velocity_path))
    options = ort.SessionOptions()
    options.intra_op_num_threads = 1
    options.inter_op_num_threads = 1
    summary_session = ort.InferenceSession(
        str(args.output_dir / "summary.onnx"),
        options,
        providers=["CPUExecutionProvider"],
    )
    velocity_session = ort.InferenceSession(
        str(velocity_path),
        options,
        providers=["CPUExecutionProvider"],
    )
    ranges = model_spec["size_ranges"]
    corners = sorted({(p, i) for p in ranges["n_persons"] for i in ranges["n_items"]})
    report = {
        "scope": "ONNX summary and velocity; Python adapter and original ODE solver",
        "checkpoint": args.model.name,
        "checkpoint_sha256": hashlib.sha256(args.model.read_bytes()).hexdigest(),
        "seed": args.seed,
        "datasets_per_corner": args.datasets,
        "draws": args.draws,
        "batch_size": args.batch_size,
        "model_family": model_family,
        "solver": flow.integrate_kwargs,
        "versions": {
            "bayesflow": bayesflow.__version__,
            "keras": keras.__version__,
            "torch": torch.__version__,
            "onnxruntime": ort.__version__,
        },
        "tolerances": {
            "metric_atol": args.metric_atol,
            "calibration_atol": args.calibration_atol,
            "draw_atol": 1e-3,
            "draw_rtol": 1e-3,
        },
        "corners": [],
        "complete": False,
        "passed": False,
    }
    failures = []
    for corner_index, (persons, items) in enumerate(corners):
        # Every recorded prior, the ability prior included, so the validation
        # population is the checkpoint's training distribution.
        simulator = irt.build_simulator(
            model_spec,
            n_items=(items, items),
            n_persons=(persons, persons),
            seed=args.seed + corner_index,
        )
        raw = simulator.sample((args.datasets,))
        native_batches = {name: [] for name in parameter_names}
        onnx_batches = {name: [] for name in parameter_names}
        noise_batches = []
        for start in range(0, args.datasets, args.batch_size):
            stop = min(start + args.batch_size, args.datasets)
            batch = {key: value[start:stop] for key, value in raw.items()}
            adapted = loaded.adapter(batch, strict=False)
            responses = np.asarray(adapted["summary_variables"], dtype="float32")
            batch_size = stop - start
            global_conditions = adapted.get("inference_conditions")
            with torch.no_grad():
                native_embedding = loaded.summary_network(
                    torch.from_numpy(responses), training=False
                )
                onnx_embedding = summary_session.run(
                    ["embeddings"], {"responses": responses}
                )[0]

                def conditions(embedding):
                    flat = loaded._flatten_conditions(
                        embedding,
                        None
                        if global_conditions is None
                        else keras.ops.convert_to_tensor(global_conditions),
                        batch_size,
                        items,
                    )
                    return flat.repeat_interleave(args.draws, dim=0)

                native_conditions = conditions(native_embedding)
                onnx_conditions = conditions(torch.from_numpy(onnx_embedding))
                # One base draw shared by the two paths removes Monte Carlo noise
                # from their comparison while honoring the checkpoint distribution.
                noise = flow.base_distribution.sample(
                    (batch_size * items * args.draws,)
                )
                native = flow(
                    noise, conditions=native_conditions, inverse=True, training=False
                )
                onnx_cond_np = onnx_conditions.detach().cpu().numpy()

                def deltas(time, xz):
                    state_np = xz.detach().cpu().numpy()
                    time_np = np.full((state_np.shape[0], 1), float(time), "float32")
                    out = velocity_session.run(
                        ["velocity"],
                        {
                            "state": state_np,
                            "time": time_np,
                            "conditions": onnx_cond_np,
                        },
                    )[0]
                    return {"xz": torch.from_numpy(out)}

                converted = integrate(
                    deltas,
                    {"xz": noise.clone()},
                    start_time=0.0,
                    stop_time=1.0,
                    **flow.integrate_kwargs,
                )["xz"]

                def inverse_adapter(samples):
                    samples = samples.reshape(batch_size, items, args.draws, param_dim)
                    samples = samples.permute(0, 2, 1, 3).detach().cpu().numpy()
                    return loaded.adapter(
                        {"inference_variables": samples}, inverse=True, strict=False
                    )

                native_out, onnx_out = (
                    inverse_adapter(native),
                    inverse_adapter(converted),
                )
                for name in parameter_names:
                    native_batches[name].append(np.asarray(native_out[name]))
                    onnx_batches[name].append(np.asarray(onnx_out[name]))
                noise_batches.append(noise.detach().cpu().numpy())
            print(
                f"corner P={persons} I={items}: datasets {stop}/{args.datasets}",
                flush=True,
            )

        parameter_results = {}
        fixtures = {
            "responses": raw["responses"],
            "base_noise": np.concatenate(noise_batches),
        }
        for parameter in parameter_names:
            native_draws = np.concatenate(native_batches[parameter])
            onnx_draws = np.concatenate(onnx_batches[parameter])
            true_values = np.asarray(raw[parameter])
            native_flat = native_draws.transpose(0, 2, 1).reshape(-1, args.draws)
            onnx_flat = onnx_draws.transpose(0, 2, 1).reshape(-1, args.draws)
            truth = true_values.reshape(-1)

            def metrics(draws):
                result = _compute_recovery_metrics(
                    truth, draws.mean(axis=1)
                ) | irt.compute_calibration_error(draws, truth)
                if parameter == "a":
                    result.update(
                        {
                            f"log_{key}": value
                            for key, value in _compute_log_recovery_metrics(
                                truth, draws
                            ).items()
                        }
                    )
                return result

            native_metrics, onnx_metrics = metrics(native_flat), metrics(onnx_flat)
            differences = {
                key: abs(native_metrics[key] - onnx_metrics[key])
                for key in native_metrics
            }
            passed = bool(
                np.isfinite(native_draws).all()
                and np.isfinite(onnx_draws).all()
                and np.allclose(native_draws, onnx_draws, atol=1e-3, rtol=1e-3)
            )
            for name, difference in differences.items():
                tolerance = (
                    args.calibration_atol
                    if name == "calibration_error"
                    else args.metric_atol
                )
                if not np.isfinite(difference) or difference > tolerance:
                    passed = False
                    failures.append(
                        f"P={persons}, I={items}, {parameter}: "
                        f"{name} delta {difference}"
                    )
            if not passed:
                failures.append(
                    f"P={persons}, I={items}, {parameter}: "
                    "posterior/metric parity failed"
                )
            parameter_results[parameter] = {
                "native": native_metrics,
                "onnx": onnx_metrics,
                "absolute_differences": differences,
                "max_draw_difference": float(np.max(np.abs(native_draws - onnx_draws))),
                "passed": passed,
            }
            fixtures.update(
                {
                    f"truth_{parameter}": true_values,
                    f"native_draws_{parameter}": native_draws,
                    f"onnx_draws_{parameter}": onnx_draws,
                }
            )
        result = {
            "persons": persons,
            "items": items,
            "parameters": parameter_results,
            "passed": all(value["passed"] for value in parameter_results.values()),
        }
        report["corners"].append(result)
        report["complete"] = len(report["corners"]) == len(corners)
        report["passed"] = report["complete"] and all(
            corner["passed"] for corner in report["corners"]
        )
        np.savez_compressed(
            args.output_dir / f"validation_p{persons}_i{items}.npz",
            **fixtures,
        )
        (args.output_dir / "recovery_report.json").write_text(
            json.dumps(report, indent=2) + "\n"
        )
        print(json.dumps(result, indent=2), flush=True)
    if failures:
        raise AssertionError("\n".join(failures))
    print(f"PASS recovery and calibration parity: {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
