"""Pytest configuration — must run before any Keras / BayesFlow imports.

NOTE: Tests currently run with the PyTorch backend only.  Keras 3 does not
support switching backends within a single process, so testing multiple
backends requires separate CI jobs (or separate ``tox`` environments).
See .github/workflows/tests.yml for the CI setup.
"""

import json
import os
import zipfile

os.environ["KERAS_BACKEND"] = "torch"

import bayesflow as bf
import pytest

from bayesflow_irt import (
    EquivariantIRTApproximator,
    Prior,
    make_1pl_adapter,
    make_1pl_simulator,
)
from bayesflow_irt.networks import MatrixEquivariantIRTSummary


def strip_registered_name_prefixes(source_path, target_path) -> None:
    """Rewrite a checkpoint's ``registered_name`` keys to their bare form.

    Checkpoints written before the classes were registered as Keras
    serializables carry the plain class name rather than
    ``"bayesflow_irt>Name"``. This reproduces such a file from a freshly saved
    one so the legacy load path can be exercised without vendoring a binary
    fixture.
    """

    def strip(node) -> None:
        if isinstance(node, dict):
            registered_name = node.get("registered_name")
            if isinstance(registered_name, str) and registered_name.startswith(
                "bayesflow_irt>"
            ):
                node["registered_name"] = registered_name.split(">", 1)[1]
            for value in node.values():
                strip(value)
        elif isinstance(node, list):
            for value in node:
                strip(value)

    with (
        zipfile.ZipFile(source_path) as source,
        zipfile.ZipFile(target_path, "w") as target,
    ):
        for info in source.infolist():
            data = source.read(info.filename)
            if info.filename == "config.json":
                config = json.loads(data)
                strip(config)
                data = json.dumps(config).encode("utf-8")
            target.writestr(info, data)


_PRIOR_B = Prior("normal", loc=0, scale=2.0)
_PRIOR_THETA = Prior("normal", loc=0, scale=1.0)


@pytest.fixture(scope="session")
def trained_1pl_approximator():
    """Build and train a small 1PL approximator for testing."""
    simulator = make_1pl_simulator(
        n_items=(10, 15),
        n_persons=(20, 30),
        prior_b=_PRIOR_B,
        prior_theta=_PRIOR_THETA,
    )
    adapter = make_1pl_adapter(simulator)

    summary_net = MatrixEquivariantIRTSummary(
        embed_dim=16,
        matrix_hidden_dim=16,
        num_matrix_blocks=1,
        num_outer_sabs=1,
        outer_embed_dim=16,
        outer_mlp_width=32,
    )
    inference_net = bf.networks.FlowMatching(
        subnet_kwargs={"widths": (32, 32)},
    )

    approximator = EquivariantIRTApproximator(
        adapter=adapter,
        inference_network=inference_net,
        item_posterior="flow_matching",
        summary_network=summary_net,
    )
    approximator.compile(optimizer="adam")

    # Train for 1 epoch just to get a working approximator
    approximator.fit(
        simulator=simulator,
        epochs=1,
        num_batches=4,
        batch_size=10,
        workers=1,
        verbose=False,
    )

    return approximator, simulator
