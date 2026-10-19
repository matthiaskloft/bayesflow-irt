"""Export an IRT summary network and verify CPU ONNX inference.

Install the project and export dependencies in a separate environment::

    python -m pip install -e ".[torch]" onnx onnxruntime onnxscript
    python scripts/test_onnx_export.py --output-dir .tmp/onnx-smoke

The default uses randomly initialized weights: this tests export, not estimation
accuracy. Pass --model path/to/model.keras to test a saved summary network or
an approximator with a summary_network attribute. Only the summary is exported;
adapters, parameter transformations and posterior sampling are out of scope.

The resulting responses.csv and embeddings.csv use persons-by-items and
items-by-features layouts respectively, for checking the artifact from R.

Implementation references:
    https://docs.pytorch.org/docs/stable/onnx.html
    https://onnxruntime.ai/docs/api/python/api_summary.html
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
from pathlib import Path


def main() -> None:
    """Export once, check several shapes, and write an R-friendly fixture."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, help="Saved .keras model (optional)")
    parser.add_argument(
        "--factory",
        type=Path,
        help="Python file with build_model() returning a built model/approximator",
    )
    parser.add_argument("--weights", type=Path, help="Weights-only .weights.h5 file")
    parser.add_argument("--output-dir", type=Path, default=Path(".tmp/onnx-smoke"))
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if args.factory and args.model:
        parser.error("Use --factory or --model, not both")
    if args.weights and not (args.factory or args.model):
        parser.error(
            "--weights requires --factory or --model; architecture is required"
        )

    # This is a standalone process; select the backend before any ML imports.
    os.environ["KERAS_BACKEND"] = "torch"
    required = ("torch", "keras", "bayesflow_irt", "onnx", "onnxruntime", "onnxscript")
    missing = [name for name in required if importlib.util.find_spec(name) is None]
    if missing:
        parser.error(
            f"Missing dependencies: {', '.join(missing)}. Run: "
            'python -m pip install -e ".[torch]" onnx onnxruntime onnxscript'
        )

    import bayesflow
    import keras
    import numpy as np
    import onnx
    import onnxruntime as ort
    import torch

    # Import registers package classes for Keras deserialization.
    import bayesflow_irt

    keras.utils.set_random_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    if args.factory:
        spec = importlib.util.spec_from_file_location(
            "onnx_model_factory", args.factory
        )
        if spec is None or spec.loader is None:
            parser.error(f"Cannot load factory: {args.factory}")
        factory = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(factory)
        loaded = factory.build_model()
        if args.weights:
            # Strict loading: never silently skip mismatched checkpoint tensors.
            loaded.load_weights(args.weights, skip_mismatch=False)
        summary = getattr(loaded, "summary_network", loaded)
    elif args.model:
        # Older checkpoints use unqualified registration names.
        custom_objects = {
            name: getattr(bayesflow_irt, name)
            for name in (
                "EquivariantIRTApproximator",
                "EquivariantIRTSummary",
                "MatrixEquivariantIRTSummary",
            )
            if hasattr(bayesflow_irt, name)
        }
        loaded = keras.saving.load_model(
            args.model, compile=False, custom_objects=custom_objects
        )
        if args.weights:
            loaded.load_weights(args.weights, skip_mismatch=False)
        summary = getattr(loaded, "summary_network", loaded)
    else:
        summary = bayesflow_irt.MatrixEquivariantIRTSummary(
            embed_dim=16,
            matrix_hidden_dim=16,
            num_matrix_blocks=1,
            person_pooling="mean",
            num_outer_sabs=0,
            outer_embed_dim=16,
            dropout=0.0,
        )

    # Build eagerly before tracing; all subsequent tests reuse these weights.
    example = torch.from_numpy(rng.integers(0, 2, (2, 24, 8)).astype("float32"))
    with torch.no_grad():
        summary(example, training=False)

    class SummaryModule(torch.nn.Module):
        """Expose only the deterministic inference forward pass to ONNX."""

        def __init__(self, layer):
            super().__init__()
            self.layer = layer

        def forward(self, responses):
            return self.layer(responses, training=False)

    module = SummaryModule(summary).eval()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    artifact = args.output_dir / "summary.onnx"
    torch.onnx.export(
        module,
        (example,),
        str(artifact),
        input_names=["responses"],
        output_names=["embeddings"],
        # Exporter progress includes Unicode glyphs unsupported by some
        # Windows consoles; parity results below provide portable output.
        verbose=False,
        dynamo=True,
        dynamic_shapes={
            "responses": {
                0: torch.export.Dim("batch", min=1),
                1: torch.export.Dim("persons", min=2),
                2: torch.export.Dim("items", min=2),
            }
        },
        external_data=False,
    )
    onnx.checker.check_model(str(artifact))
    session = ort.InferenceSession(str(artifact), providers=["CPUExecutionProvider"])
    results = []
    for shape in ((2, 24, 8), (1, 13, 5), (3, 31, 11)):
        responses = rng.integers(0, 2, shape).astype("float32")
        with torch.no_grad():
            expected = module(torch.from_numpy(responses)).detach().cpu().numpy()
        actual = session.run(["embeddings"], {"responses": responses})[0]
        if actual.shape != expected.shape:
            raise AssertionError(
                f"Output shapes differ: {actual.shape}, {expected.shape}"
            )
        np.testing.assert_allclose(actual, expected, rtol=1e-4, atol=1e-5)
        if not np.isfinite(actual).all():
            raise AssertionError("ONNX produced non-finite outputs")
        error = float(np.max(np.abs(actual - expected)))
        print(f"PASS input={shape}, output={actual.shape}, max_abs_error={error:.3g}")
        results.append(
            {"input_shape": shape, "output_shape": actual.shape, "error": error}
        )
        if shape[0] == 1:
            np.savetxt(args.output_dir / "responses.csv", responses[0], delimiter=",")
            np.savetxt(args.output_dir / "embeddings.csv", actual[0], delimiter=",")

    report = {
        "scope": "summary network only; not posterior estimation",
        "weights": (
            str(args.weights or args.model or args.factory)
            if (args.weights or args.model or args.factory)
            else "random initialization"
        ),
        "versions": {
            "torch": torch.__version__,
            "keras": keras.__version__,
            "bayesflow": bayesflow.__version__,
            "onnx": onnx.__version__,
            "onnxruntime": ort.__version__,
        },
        "tests": results,
    }
    (args.output_dir / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(f"Export and parity checks passed. Files: {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
