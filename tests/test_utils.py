"""Tests for bayesflow_irt.utils."""

import builtins

import keras
import numpy as np
import pytest

import bayesflow_irt.utils as utils_mod
from bayesflow_irt.networks import MatrixEquivariantIRTSummary
from bayesflow_irt.utils import (
    ChunkSearchOptions,
    InferenceProfile,
    MemoryProbeReport,
    ProbeResult,
    TrainingMemoryConfig,
    configure_large_scale_training,
    fit_with_oom_retry,
    is_oom_error,
    probe_memory_config,
    profile_posterior_inference,
    safe_empty_cache,
    set_summary_memory_config,
)

# ---------------------------------------------------------------------------
# Training-memory helpers
# ---------------------------------------------------------------------------


class TestIsOomError:
    def test_cuda_oom(self):
        exc = RuntimeError("CUDA out of memory. Tried to allocate 2.00 GiB")
        assert is_oom_error(exc) is True

    def test_cuda_alloc_error(self):
        exc = RuntimeError("cudaErrorMemoryAllocation: out of memory")
        assert is_oom_error(exc) is True

    def test_torch_cpu_allocation_failure(self):
        """The CPU allocator's message has no "out of memory" in it."""
        import torch

        try:
            torch.empty(2**50, dtype=torch.uint8)
        except RuntimeError as exc:
            assert "DefaultCPUAllocator" in str(exc)
            assert is_oom_error(exc) is True
        else:  # pragma: no cover - a petabyte allocation succeeded
            pytest.skip("allocation unexpectedly succeeded")

    def test_linux_cpu_allocator_message(self):
        exc = RuntimeError(
            "[enforce fail at alloc_cpu.cpp:127] err == 0. DefaultCPUAllocator: "
            "can't allocate memory: you tried to allocate 1125899906842624 bytes."
        )
        assert is_oom_error(exc) is True

    def test_non_oom(self):
        exc = RuntimeError("some other error")
        assert is_oom_error(exc) is False

    def test_value_error(self):
        # Backend-aware detection (torch backend here) only treats
        # RuntimeError-family exceptions as OOM, so an unrelated ValueError
        # that merely mentions "out of memory" in its message must not match.
        exc = ValueError("out of memory somewhere")
        assert is_oom_error(exc) is False

    def test_torch_oom_error_type(self):
        import torch

        exc = torch.cuda.OutOfMemoryError("CUDA out of memory")
        assert is_oom_error(exc) is True

    def test_torch_backward_engine_system_error(self):
        """A genuine CUDA allocation failure inside torch's C++ autograd
        engine (e.g. during _ChunkedTrainableMatrixPool.backward's per-chunk
        recompute) can surface on the calling thread as a bare SystemError
        with CPython's generic C-API-failure message rather than a typed
        torch.cuda.OutOfMemoryError -- this can surface on some platforms.
        Must be retried like any other OOM."""
        exc = SystemError(
            "<built-in method run_backward> returned a result with an error set"
        )
        assert is_oom_error(exc) is False  # wrong message: not the OOM pattern
        exc = SystemError("error return without exception set")
        assert is_oom_error(exc) is True

    def test_unrelated_system_error_not_treated_as_oom(self):
        exc = SystemError("some unrelated interpreter-level bug")
        assert is_oom_error(exc) is False


class TestSafeEmptyCache:
    def test_does_not_raise(self):
        """Should never raise regardless of CUDA availability."""
        safe_empty_cache()


class TestCapCudaMemoryFraction:
    """Tests for _cap_cuda_memory_fraction, the WDDM-spillover guard.

    See its docstring: on Windows, an over-budget CUDA allocation can be
    silently satisfied by the driver spilling into slow shared/system
    memory instead of raising torch.cuda.OutOfMemoryError. Capping the
    allocator via torch.cuda.set_per_process_memory_fraction converts that
    into a real, fast, catchable OOM.
    """

    def setup_method(self):
        # Each test starts from a clean idempotency-guard state, regardless
        # of what earlier tests (or a real CUDA device on this machine)
        # already recorded.
        utils_mod._cuda_memory_fraction_capped.clear()

    def test_applies_cap_when_cuda_available(self, monkeypatch):
        """An untouched device (no prior limit, default fraction 1.0) gets
        tightened to *fraction*."""
        calls = []
        monkeypatch.setattr("torch.cuda.is_available", lambda: True)
        monkeypatch.setattr("torch.cuda.current_device", lambda: 0)
        monkeypatch.setattr(
            "torch.cuda.get_per_process_memory_fraction", lambda device: 1.0
        )
        monkeypatch.setattr(
            "torch.cuda.set_per_process_memory_fraction",
            lambda fraction, device: calls.append((fraction, device)),
        )

        utils_mod._cap_cuda_memory_fraction(0.95)

        assert calls == [(0.95, 0)]
        assert utils_mod._cuda_memory_fraction_capped == {0}

    def test_never_loosens_stricter_existing_limit(self, monkeypatch):
        """If an embedding application already capped this device tighter
        (e.g. 0.5, to share one GPU across processes), this must not
        override it down to a looser ~95%.
        """
        calls = []
        monkeypatch.setattr("torch.cuda.is_available", lambda: True)
        monkeypatch.setattr("torch.cuda.current_device", lambda: 0)
        monkeypatch.setattr(
            "torch.cuda.get_per_process_memory_fraction", lambda device: 0.5
        )
        monkeypatch.setattr(
            "torch.cuda.set_per_process_memory_fraction",
            lambda fraction, device: calls.append((fraction, device)),
        )

        utils_mod._cap_cuda_memory_fraction(0.95)

        assert calls == []  # 0.95 is looser than the existing 0.5 -- left alone
        assert utils_mod._cuda_memory_fraction_capped == {0}

    def test_noop_when_cuda_unavailable(self, monkeypatch):
        calls = []
        monkeypatch.setattr("torch.cuda.is_available", lambda: False)
        monkeypatch.setattr(
            "torch.cuda.set_per_process_memory_fraction",
            lambda fraction, device: calls.append((fraction, device)),
        )

        utils_mod._cap_cuda_memory_fraction(0.95)

        assert calls == []
        assert utils_mod._cuda_memory_fraction_capped == set()

    def test_idempotent_per_device(self, monkeypatch):
        """A second call on an already-capped device is a cheap no-op.

        Both probe_memory_config and fit_with_oom_retry call this at their
        own start, so whichever runs first in a process must not have its
        cap silently loosened (or the allocator needlessly re-touched) by
        the other.
        """
        calls = []
        monkeypatch.setattr("torch.cuda.is_available", lambda: True)
        monkeypatch.setattr("torch.cuda.current_device", lambda: 0)
        monkeypatch.setattr(
            "torch.cuda.get_per_process_memory_fraction", lambda device: 1.0
        )
        monkeypatch.setattr(
            "torch.cuda.set_per_process_memory_fraction",
            lambda fraction, device: calls.append((fraction, device)),
        )

        utils_mod._cap_cuda_memory_fraction(0.95)
        utils_mod._cap_cuda_memory_fraction(0.5)  # looser fraction, still ignored

        assert calls == [(0.95, 0)]

    def test_swallows_runtime_error(self, monkeypatch):
        """A CUDA-unavailable-at-the-driver-level RuntimeError must not raise.

        This is the actual behavior relied on by probe_memory_config /
        fit_with_oom_retry tests that mock torch.cuda.is_available() to
        True on machines without a real CUDA device: current_device() would
        otherwise raise, and this helper must degrade to a no-op instead of
        failing the caller.
        """
        monkeypatch.setattr("torch.cuda.is_available", lambda: True)

        def _raise(*_args, **_kwargs):
            raise RuntimeError("No CUDA GPUs are available")

        monkeypatch.setattr("torch.cuda.current_device", _raise)

        utils_mod._cap_cuda_memory_fraction(0.95)  # must not raise

        assert utils_mod._cuda_memory_fraction_capped == set()


class TestSetSummaryMemoryConfig:
    def test_sets_attributes(self):
        class FakeNet:
            item_chunk_size = None
            gradient_checkpointing = False
            _config = {"item_chunk_size": None, "gradient_checkpointing": False}

        net = FakeNet()
        set_summary_memory_config(net, item_chunk_size=32, gradient_checkpointing=True)
        assert net.item_chunk_size == 32
        assert net.gradient_checkpointing is True
        assert net._config["item_chunk_size"] == 32
        assert net._config["gradient_checkpointing"] is True

    def test_no_config_dict(self):
        """Should work even if _config is missing."""

        class Bare:
            item_chunk_size = None
            gradient_checkpointing = False

        net = Bare()
        set_summary_memory_config(net, item_chunk_size=16, gradient_checkpointing=False)
        assert net.item_chunk_size == 16

    @pytest.mark.parametrize(
        "summary_net",
        [
            MatrixEquivariantIRTSummary(
                embed_dim=8, matrix_hidden_dim=8, num_matrix_blocks=1
            ),
        ],
    )
    def test_real_summary_classes(self, summary_net):
        """Summary classes should expose the memory-config contract."""
        set_summary_memory_config(
            summary_net, item_chunk_size=32, gradient_checkpointing=True
        )
        assert summary_net.item_chunk_size == 32
        assert summary_net.gradient_checkpointing is True
        assert summary_net.get_config()["item_chunk_size"] == 32
        assert summary_net.get_config()["gradient_checkpointing"] is True

    def test_item_chunking_capability_contract(self):
        assert MatrixEquivariantIRTSummary.supports_item_chunking is False
        assert not utils_mod._supports_item_chunking(
            MatrixEquivariantIRTSummary(embed_dim=8)
        )

    def test_sets_person_chunk_size_when_attribute_present(self):
        class FakeNet:
            item_chunk_size = None
            gradient_checkpointing = False
            person_chunk_size = None
            _config = {
                "item_chunk_size": None,
                "gradient_checkpointing": False,
                "person_chunk_size": None,
            }

        net = FakeNet()
        set_summary_memory_config(
            net,
            item_chunk_size=None,
            gradient_checkpointing=True,
            person_chunk_size=64,
        )
        assert net.person_chunk_size == 64
        assert net._config["person_chunk_size"] == 64

    def test_ignores_person_chunk_size_when_attribute_absent(self):
        """A summary class predating person chunking has no attribute to set."""

        class FakeNet:
            item_chunk_size = None
            gradient_checkpointing = False
            _config = {"item_chunk_size": None, "gradient_checkpointing": False}

        net = FakeNet()
        set_summary_memory_config(
            net,
            item_chunk_size=None,
            gradient_checkpointing=True,
            person_chunk_size=64,
        )
        assert not hasattr(net, "person_chunk_size")
        assert "person_chunk_size" not in net._config

    def test_real_matrix_summary_person_chunk_size(self):
        summary_net = MatrixEquivariantIRTSummary(
            embed_dim=8, matrix_hidden_dim=8, num_matrix_blocks=1
        )
        set_summary_memory_config(
            summary_net,
            item_chunk_size=None,
            gradient_checkpointing=False,
            person_chunk_size=64,
        )
        assert summary_net.person_chunk_size == 64
        assert summary_net.get_config()["person_chunk_size"] == 64

    def test_person_chunking_capability_contract(self):
        assert MatrixEquivariantIRTSummary.supports_person_chunking is True
        assert utils_mod._supports_person_chunking(
            MatrixEquivariantIRTSummary(embed_dim=8)
        )
        assert not utils_mod._supports_person_chunking(None)

        class FakeHasAttrNoFlag:
            # hasattr alone isn't sufficient for person chunking (unlike item
            # chunking's backward-compatible default): a class must opt in
            # explicitly via supports_person_chunking = True.
            person_chunk_size = None

        assert not utils_mod._supports_person_chunking(FakeHasAttrNoFlag())

    def test_person_chunking_unsupported_for_pma_pooling_instance(self):
        """A PMA-pooling instance is not person-chunking-eligible even
        though the class-level flag is True: __init__ only validates
        person_pooling=="mean" against person_chunk_size when the latter is
        set at construction time, so a PMA instance with person_chunk_size
        left at its default None still reports the class flag as True. The
        instance-level _person_chunking_config_is_eligible() check must
        catch this so set_summary_memory_config (via probe/retry) never
        mutates person_chunk_size on an instance where the resulting
        network.call() couldn't safely honor it."""
        summary_net = MatrixEquivariantIRTSummary(
            embed_dim=8,
            matrix_hidden_dim=8,
            num_matrix_blocks=1,
            person_pooling="pma",
        )
        assert summary_net.supports_person_chunking is True
        assert not utils_mod._supports_person_chunking(summary_net)

    def test_person_chunking_unsupported_for_cell_level_raw_features(self):
        summary_net = MatrixEquivariantIRTSummary(
            embed_dim=8,
            matrix_hidden_dim=8,
            num_matrix_blocks=1,
            raw_feature_channels=("rest_mean_score",),
        )
        assert not utils_mod._supports_person_chunking(summary_net)


class TestProbeResult:
    def test_peak_mib(self):
        r = ProbeResult(
            item_chunk_size=32,
            gradient_checkpointing=False,
            ok=True,
            peak_allocated_bytes=2**30,
            peak_reserved_bytes=2**30,
        )
        assert r.peak_mib == 1024.0
        assert r.peak_reserved_mib == 1024.0

    def test_peak_mib_none_on_oom(self):
        r = ProbeResult(item_chunk_size=32, gradient_checkpointing=False, ok=False)
        assert r.peak_mib is None

    def test_person_chunk_size_defaults_to_none(self):
        r = ProbeResult(item_chunk_size=None, gradient_checkpointing=False, ok=True)
        assert r.person_chunk_size is None

    def test_person_chunk_size_round_trips(self):
        r = ProbeResult(
            item_chunk_size=None,
            gradient_checkpointing=False,
            ok=True,
            person_chunk_size=128,
        )
        assert r.person_chunk_size == 128


class TestFormatProbeRow:
    """Direct unit tests for utils_mod._format_probe_row.

    Integration tests through probe_memory_config /
    _search_chunk_size_boundary mostly run with verbose=False, so without
    these a rename or formatting regression on either the item= or reason=
    field could pass all tests silently.
    """

    def test_item_field_uses_item_prefix_not_chunk(self):
        """The chunk-naming rename (chunk= -> item=) must hold on the item
        side too, not just person= (see test_summary_lines_show_person_chunk_size)."""
        r = ProbeResult(
            item_chunk_size=64,
            gradient_checkpointing=False,
            ok=True,
            peak_allocated_bytes=2**20,
            peak_reserved_bytes=2**20,
        )
        line = utils_mod._format_probe_row(r)
        assert "item=" in line
        assert "64" in line
        assert "chunk=" not in line

    def test_reason_suffix_only_on_failed_rows(self):
        ok_row = utils_mod._format_probe_row(
            ProbeResult(item_chunk_size=32, gradient_checkpointing=False, ok=True)
        )
        assert "reason=" not in ok_row

        fail_row = utils_mod._format_probe_row(
            ProbeResult(
                item_chunk_size=32,
                gradient_checkpointing=False,
                ok=False,
                reason="oom",
            )
        )
        assert "reason=oom" in fail_row

    def test_note_is_appended(self):
        r = ProbeResult(item_chunk_size=8, gradient_checkpointing=True, ok=True)
        line = utils_mod._format_probe_row(r, note="dense baseline")
        assert "dense baseline" in line

    def test_peak_reserved_pct_rendered_when_device_total_set(self):
        r = ProbeResult(
            item_chunk_size=8,
            gradient_checkpointing=False,
            ok=True,
            peak_reserved_bytes=2**29,
            device_total_bytes=2**30,
        )
        line = utils_mod._format_probe_row(r)
        assert "50.0%" in line

    def test_peak_reserved_pct_omitted_without_device_total(self):
        r = ProbeResult(
            item_chunk_size=8,
            gradient_checkpointing=False,
            ok=True,
            peak_reserved_bytes=2**29,
        )
        line = utils_mod._format_probe_row(r)
        assert "%" not in line


class TestMemoryProbeReport:
    def test_summary_lines(self):
        report = MemoryProbeReport(batch_size=64, n_items=20, n_persons=100)
        report.trials.append(
            ProbeResult(
                item_chunk_size=None,
                gradient_checkpointing=False,
                ok=True,
                peak_allocated_bytes=2**30,
                peak_reserved_bytes=2**30,
            )
        )
        report.best = report.trials[0]
        lines = report.summary_lines()
        assert any("batch=64" in line for line in lines)
        assert any("best" in line for line in lines)

    def test_no_best(self):
        report = MemoryProbeReport(batch_size=64, n_items=20, n_persons=100)
        lines = report.summary_lines()
        assert any("no config" in line for line in lines)

    def test_summary_lines_show_person_chunk_size(self):
        report = MemoryProbeReport(batch_size=64, n_items=20, n_persons=100)
        report.trials.append(
            ProbeResult(
                item_chunk_size=None,
                gradient_checkpointing=False,
                ok=True,
                peak_allocated_bytes=2**30,
                peak_reserved_bytes=2**30,
                person_chunk_size=256,
            )
        )
        report.best = report.trials[0]
        lines = report.summary_lines()
        assert any("person=" in line and "256" in line for line in lines)


class TestTrainingMemoryConfigValidation:
    def test_accepts_valid_defaults(self):
        cfg = TrainingMemoryConfig()
        assert cfg.gradient_accumulation_steps == 1

    def test_normalizes_legacy_flow_chunk_backward_none(self):
        cfg = TrainingMemoryConfig(flow_chunk_backward="none")
        assert cfg.flow_chunk_backward is None

    @pytest.mark.parametrize(
        "kwargs",
        [
            {"item_chunk_size": 0},
            {"flow_chunk_size": -1},
            {"flow_chunk_backward": "banana"},
            {"gradient_accumulation_steps": 0},
            {"person_chunk_size": 0},
            {"person_chunk_size": -1},
        ],
    )
    def test_rejects_invalid_values(self, kwargs):
        with pytest.raises(ValueError):
            TrainingMemoryConfig(**kwargs)

    def test_accepts_valid_person_chunk_size(self):
        cfg = TrainingMemoryConfig(person_chunk_size=64)
        assert cfg.person_chunk_size == 64


class TestInferenceProfile:
    def test_profile_posterior_inference_breaks_down_sampling_path(self):
        class FakeAdapter:
            def __init__(self):
                self.forward_calls = []
                self.inverse_calls = []

            def __call__(self, data, strict=True, inverse=False):
                if inverse:
                    self.inverse_calls.append((data, strict))
                    return {"b": np.asarray(data["inference_variables"])[..., 0]}

                self.forward_calls.append((data, strict))
                if strict is False and not self.forward_calls[:-1]:
                    raise ValueError(
                        "Cannot call `forward` with `strict=False` before "
                        "calling `forward` with `strict=True`."
                    )
                return {
                    "summary_variables": np.asarray(
                        data["responses"], dtype=np.float32
                    ),
                    "inference_conditions": np.asarray(data["meta"], dtype=np.float32),
                }

        class FakeSummary:
            def __call__(self, x):
                batch, _, items = x.shape
                return np.zeros((batch, items, 3), dtype=np.float32)

        class FakeInference:
            def __init__(self):
                self.calls = []

            def sample(self, batch_shape, conditions=None, **kwargs):
                cond_shape = tuple(
                    keras.ops.convert_to_numpy(keras.ops.shape(conditions))
                )
                self.calls.append((batch_shape, cond_shape, kwargs))
                n_cond, n_sample = batch_shape
                return np.zeros((n_cond, n_sample, 1), dtype=np.float32)

        class FakeApprox:
            def __init__(self):
                self.adapter = FakeAdapter()
                self.summary_network = FakeSummary()
                self.inference_network = FakeInference()
                self.standardize_params = False
                self._global_condition_dim = 2

            @staticmethod
            def _flatten_conditions(
                summary_outputs, inference_conditions, batch_size, num_items
            ):
                flat_summary = keras.ops.reshape(
                    summary_outputs, (batch_size * num_items, -1)
                )
                expanded = keras.ops.broadcast_to(
                    inference_conditions[:, None, :],
                    (batch_size, num_items, inference_conditions.shape[-1]),
                )
                flat_global = keras.ops.reshape(expanded, (batch_size * num_items, -1))
                return keras.ops.concatenate([flat_summary, flat_global], axis=-1)

        approx = FakeApprox()
        conditions = {
            "responses": np.zeros((2, 5, 3), dtype=np.float32),
            "meta": np.ones((2, 2), dtype=np.float32),
        }

        profile = profile_posterior_inference(
            approx,
            conditions,
            num_samples=5,
            sample_chunk_size=2,
            condition_chunk_size=4,
            flow_sample_kwargs={"method": "euler", "steps": 24},
        )

        assert isinstance(profile, InferenceProfile)
        assert profile.batch_size == 2
        assert profile.num_items == 3
        assert profile.flat_conditions == 6
        assert profile.num_sample_chunks == 3
        assert profile.num_condition_chunks == 2
        assert profile.inference_sample_calls == 6
        assert profile.output_shapes == {"b": (2, 5, 3)}
        assert len(profile.summary_lines()) == 3
        assert len(approx.inference_network.calls) == 6
        assert all(
            call[2] == {"method": "euler", "steps": 24}
            for call in approx.inference_network.calls
        )
        assert approx.adapter.forward_calls[0][1] is False
        assert approx.adapter.forward_calls[1][1] is True
        assert approx.adapter.forward_calls[2][1] is False
        assert approx.adapter.inverse_calls[0][1] is False


class TestProbeMemoryConfig:
    """Tests for probe_memory_config using the disposable-clone approach."""

    @staticmethod
    def _make_fake_approx():
        class FakeSummaryNet:
            item_chunk_size = None
            gradient_checkpointing = False
            _config = {"item_chunk_size": None, "gradient_checkpointing": False}

        class FakeApprox:
            def __init__(self):
                self.summary_network = FakeSummaryNet()

        return FakeApprox()

    def test_basic_probe(self, monkeypatch):
        """Probe should clone, run trials, and apply best config."""
        monkeypatch.setattr("torch.cuda.is_available", lambda: True)
        monkeypatch.setattr(
            utils_mod,
            "_build_probe_approximator",
            lambda approx: self._make_fake_approx(),
        )
        monkeypatch.setattr(
            utils_mod,
            "_probe_one",
            lambda *args, **kwargs: ProbeResult(
                None,
                False,
                ok=True,
                peak_allocated_bytes=2**20,
                peak_reserved_bytes=2**20,
            ),
        )

        original = self._make_fake_approx()
        report = probe_memory_config(
            original,
            lambda **kwargs: None,
            batch_size=8,
            n_items=10,
            n_persons=20,
            search=ChunkSearchOptions(
                candidate_item_chunk_sizes=[None], checkpoint_options=[False]
            ),
            verbose=False,
        )

        assert report.best is not None
        assert report.best.ok is True
        # Best config applied to the ORIGINAL summary_net
        assert original.summary_network.item_chunk_size is None
        assert original.summary_network.gradient_checkpointing is False

    def test_clone_failure_raises_without_force_fallback(self, monkeypatch):
        """Clone failures should raise by default (no silent fallback)."""
        monkeypatch.setattr("torch.cuda.is_available", lambda: True)
        monkeypatch.setattr(
            utils_mod,
            "_build_probe_approximator",
            lambda approx: (_ for _ in ()).throw(
                RuntimeError("serialisation not available")
            ),
        )

        with pytest.raises(RuntimeError, match="clone failed"):
            probe_memory_config(
                self._make_fake_approx(),
                lambda **kwargs: None,
                batch_size=8,
                n_items=10,
                n_persons=20,
                verbose=False,
            )

    def test_all_oom_applies_fallback(self, monkeypatch):
        """When every trial OOMs, apply the safe fallback config."""
        monkeypatch.setattr("torch.cuda.is_available", lambda: True)
        monkeypatch.setattr(
            utils_mod,
            "_build_probe_approximator",
            lambda approx: self._make_fake_approx(),
        )
        monkeypatch.setattr(
            utils_mod,
            "_probe_one",
            lambda *args, **kwargs: ProbeResult(None, False, ok=False),
        )

        original = self._make_fake_approx()
        report = probe_memory_config(
            original,
            lambda **kwargs: None,
            batch_size=8,
            n_items=10,
            n_persons=20,
            search=ChunkSearchOptions(
                candidate_item_chunk_sizes=[None], checkpoint_options=[False]
            ),
            force_fallback=True,
            verbose=False,
        )

        assert report.best is None
        # Fallback config applied to original
        assert original.summary_network.item_chunk_size == 8
        assert original.summary_network.gradient_checkpointing is True

    def test_clone_failure_can_force_fallback(self, monkeypatch):
        monkeypatch.setattr("torch.cuda.is_available", lambda: True)
        monkeypatch.setattr(
            utils_mod,
            "_build_probe_approximator",
            lambda approx: (_ for _ in ()).throw(
                RuntimeError("serialisation not available")
            ),
        )

        original = self._make_fake_approx()
        report = probe_memory_config(
            original,
            lambda **kwargs: None,
            batch_size=8,
            n_items=10,
            n_persons=20,
            force_fallback=True,
            verbose=False,
        )

        assert report.probe_failed is True
        assert "clone failed" in (report.probe_error or "")
        assert original.summary_network.item_chunk_size == 8
        assert original.summary_network.gradient_checkpointing is True

    def test_probe_ignores_chunks_when_summary_opts_out(self, monkeypatch):
        """Non-chunking summaries should not produce meaningless chunk trials."""
        monkeypatch.setattr("torch.cuda.is_available", lambda: True)

        class FakeSummaryNet:
            supports_item_chunking = False
            item_chunk_size = None
            gradient_checkpointing = False
            _config = {"item_chunk_size": None, "gradient_checkpointing": False}

        class FakeApprox:
            def __init__(self):
                self.summary_network = FakeSummaryNet()

        seen_chunks = []

        monkeypatch.setattr(
            utils_mod,
            "_build_probe_approximator",
            lambda approx: FakeApprox(),
        )

        def fake_probe(*args, **kwargs):
            seen_chunks.append(args[3])
            return ProbeResult(args[3], args[4], ok=False)

        monkeypatch.setattr(utils_mod, "_probe_one", fake_probe)

        original = FakeApprox()
        report = probe_memory_config(
            original,
            lambda **kwargs: None,
            batch_size=8,
            n_items=10,
            n_persons=20,
            search=ChunkSearchOptions(
                candidate_item_chunk_sizes=[None, 8, 4], checkpoint_options=[False]
            ),
            force_fallback=True,
            verbose=False,
        )

        assert report.best is None
        assert seen_chunks == [None]
        assert original.summary_network.item_chunk_size is None
        assert original.summary_network.gradient_checkpointing is True

    def test_probe_ignores_person_chunks_when_summary_does_not_support(
        self, monkeypatch
    ):
        """Summaries without person-chunking support should not produce
        meaningless person-chunk trials."""
        monkeypatch.setattr("torch.cuda.is_available", lambda: True)

        seen_person_chunks = []

        monkeypatch.setattr(
            utils_mod,
            "_build_probe_approximator",
            lambda approx: self._make_fake_approx(),
        )

        def fake_probe(*args, **kwargs):
            seen_person_chunks.append(kwargs.get("person_chunk_size"))
            return ProbeResult(args[3], args[4], ok=False)

        monkeypatch.setattr(utils_mod, "_probe_one", fake_probe)

        original = self._make_fake_approx()
        report = probe_memory_config(
            original,
            lambda **kwargs: None,
            batch_size=8,
            n_items=10,
            n_persons=20,
            search=ChunkSearchOptions(
                candidate_item_chunk_sizes=[None],
                checkpoint_options=[False],
                candidate_person_chunk_sizes=[None, 16, 8],
            ),
            force_fallback=True,
            verbose=False,
        )

        assert report.best is None
        assert seen_person_chunks == [None]

    def test_probe_explores_person_chunk_sizes_for_supporting_summary(
        self, monkeypatch
    ):
        """Summaries that support person chunking get a person-chunk grid."""
        monkeypatch.setattr("torch.cuda.is_available", lambda: True)

        class FakeSummaryNet:
            supports_person_chunking = True
            item_chunk_size = None
            person_chunk_size = None
            gradient_checkpointing = False
            _config = {
                "item_chunk_size": None,
                "gradient_checkpointing": False,
                "person_chunk_size": None,
            }

        class FakeApprox:
            def __init__(self):
                self.summary_network = FakeSummaryNet()

        seen_person_chunks = []

        monkeypatch.setattr(
            utils_mod, "_build_probe_approximator", lambda approx: FakeApprox()
        )

        def fake_probe(*args, **kwargs):
            seen_person_chunks.append(kwargs.get("person_chunk_size"))
            return ProbeResult(args[3], args[4], ok=False)

        monkeypatch.setattr(utils_mod, "_probe_one", fake_probe)

        original = FakeApprox()
        report = probe_memory_config(
            original,
            lambda **kwargs: None,
            batch_size=8,
            n_items=10,
            n_persons=20,
            search=ChunkSearchOptions(
                candidate_item_chunk_sizes=[None],
                checkpoint_options=[False],
                candidate_person_chunk_sizes=[None, 16, 8],
            ),
            force_fallback=True,
            verbose=False,
        )

        assert report.best is None
        assert seen_person_chunks == [None, 16, 8]
        # Fallback applies person_chunk_size=8 since this summary supports it.
        assert original.summary_network.person_chunk_size == 8

    def test_probe_default_person_chunk_ladder_derives_from_n_persons(
        self, monkeypatch
    ):
        monkeypatch.setattr("torch.cuda.is_available", lambda: True)

        class FakeSummaryNet:
            supports_person_chunking = True
            item_chunk_size = None
            person_chunk_size = None
            gradient_checkpointing = False
            _config = {
                "item_chunk_size": None,
                "gradient_checkpointing": False,
                "person_chunk_size": None,
            }

        class FakeApprox:
            def __init__(self):
                self.summary_network = FakeSummaryNet()

        seen_person_chunks = []

        monkeypatch.setattr(
            utils_mod, "_build_probe_approximator", lambda approx: FakeApprox()
        )

        def fake_probe(*args, **kwargs):
            seen_person_chunks.append(kwargs.get("person_chunk_size"))
            return ProbeResult(args[3], args[4], ok=False)

        monkeypatch.setattr(utils_mod, "_probe_one", fake_probe)

        report = probe_memory_config(
            FakeApprox(),
            lambda **kwargs: None,
            batch_size=8,
            n_items=10,
            n_persons=32,
            search=ChunkSearchOptions(
                candidate_item_chunk_sizes=[None], checkpoint_options=[False]
            ),
            # Forces the fixed-grid scan (this test's actual target) instead
            # of the exponential+binary boundary search, which is what a
            # default-candidates + early_stop=True call runs -- see
            # test_probe_person_chunk_search_finds_exact_boundary below.
            early_stop=False,
            force_fallback=True,
            verbose=False,
        )

        assert report.best is None
        assert seen_person_chunks == [None, 32, 16, 8]

    def test_probe_person_chunk_search_finds_exact_boundary(self, monkeypatch):
        """Default candidates + early_stop=True should run the binary
        boundary search, not the fixed power-of-2 ladder, and land on
        the exact largest feasible person_chunk_size rather than a coarser
        ladder rung."""
        monkeypatch.setattr("torch.cuda.is_available", lambda: True)

        class FakeSummaryNet:
            supports_item_chunking = False
            supports_person_chunking = True
            item_chunk_size = None
            person_chunk_size = None
            gradient_checkpointing = False
            _config = {
                "item_chunk_size": None,
                "gradient_checkpointing": False,
                "person_chunk_size": None,
            }

        class FakeApprox:
            def __init__(self):
                self.summary_network = FakeSummaryNet()

        monkeypatch.setattr(
            utils_mod, "_build_probe_approximator", lambda approx: FakeApprox()
        )

        seen_person_chunks = []

        def fake_probe(*args, **kwargs):
            pcs = kwargs.get("person_chunk_size")
            seen_person_chunks.append(pcs)
            # Feasible iff chunked (pcs is not None) and pcs <= 37; dense
            # (pcs=None) is infeasible, forcing the search to actually run.
            ok = pcs is not None and pcs <= 37
            return ProbeResult(args[3], args[4], ok=ok, person_chunk_size=pcs)

        monkeypatch.setattr(utils_mod, "_probe_one", fake_probe)

        report = probe_memory_config(
            FakeApprox(),
            lambda **kwargs: None,
            batch_size=8,
            n_items=10,
            n_persons=100,
            search=ChunkSearchOptions(checkpoint_options=[False]),
            verbose=False,
        )

        assert report.best is not None
        # The exact boundary (37), not a coarser power-of-2 ladder rung (the
        # fixed grid from n_persons=100 would have stopped at 25, the first
        # rung <= 37 it happens to hit -- see the sibling ladder test above).
        assert report.best.person_chunk_size == 37
        # Dense (None, infeasible) + floor (8, feasible) + a direct binary
        # search between floor=8 and ceiling=100 (ceiling assumed infeasible
        # from the dense failure, so it costs no probe of its own) down to
        # the exact integer boundary.
        assert seen_person_chunks == [None, 8, 54, 31, 42, 36, 39, 37, 38]

    def test_probe_search_tolerance_trades_precision_for_fewer_trials(
        self, monkeypatch
    ):
        """A relative search_tolerance should stop the binary search early,
        landing on a conservative (<= true boundary) value in fewer trials
        than the default exact-boundary tolerance=1."""
        monkeypatch.setattr("torch.cuda.is_available", lambda: True)

        class FakeSummaryNet:
            supports_item_chunking = False
            supports_person_chunking = True
            item_chunk_size = None
            person_chunk_size = None
            gradient_checkpointing = False
            _config = {
                "item_chunk_size": None,
                "gradient_checkpointing": False,
                "person_chunk_size": None,
            }

        class FakeApprox:
            def __init__(self):
                self.summary_network = FakeSummaryNet()

        monkeypatch.setattr(
            utils_mod, "_build_probe_approximator", lambda approx: FakeApprox()
        )

        seen_person_chunks = []

        def fake_probe(*args, **kwargs):
            pcs = kwargs.get("person_chunk_size")
            seen_person_chunks.append(pcs)
            ok = pcs is not None and pcs <= 37
            return ProbeResult(args[3], args[4], ok=ok, person_chunk_size=pcs)

        monkeypatch.setattr(utils_mod, "_probe_one", fake_probe)

        report = probe_memory_config(
            FakeApprox(),
            lambda **kwargs: None,
            batch_size=8,
            n_items=10,
            n_persons=100,
            search=ChunkSearchOptions(checkpoint_options=[False]),
            search_tolerance=0.1,  # 10% of ceiling=100 -> abs tolerance 10
            verbose=False,
        )

        assert report.best is not None
        # Conservative: at or below the true boundary (37), never above it.
        assert report.best.person_chunk_size <= 37
        assert report.best.person_chunk_size == 36
        assert seen_person_chunks == [None, 8, 54, 31, 42, 36]
        assert len(seen_person_chunks) < 8  # fewer trials than tolerance=1

    def test_resolve_search_tolerance(self):
        assert utils_mod._resolve_search_tolerance(1, ceiling=100) == 1
        assert utils_mod._resolve_search_tolerance(5, ceiling=100) == 5
        assert utils_mod._resolve_search_tolerance(0.1, ceiling=100) == 10
        # Rounds down to a minimum of 1, never 0.
        assert utils_mod._resolve_search_tolerance(0.001, ceiling=10) == 1

        with pytest.raises(ValueError, match="must lie in"):
            utils_mod._resolve_search_tolerance(1.5, ceiling=100)
        with pytest.raises(ValueError, match="must lie in"):
            utils_mod._resolve_search_tolerance(0.0, ceiling=100)
        with pytest.raises(ValueError, match="must be >= 1"):
            utils_mod._resolve_search_tolerance(0, ceiling=100)

    def test_probe_one_builds_clone_before_train_step(self, monkeypatch):
        """_probe_one should build an unbuilt clone from concrete batch shapes."""

        monkeypatch.setattr("torch.cuda.max_memory_allocated", lambda: 0)
        monkeypatch.setattr("torch.cuda.max_memory_reserved", lambda: 0)
        monkeypatch.setattr("torch.cuda.reset_peak_memory_stats", lambda: None)
        monkeypatch.setattr("torch.cuda.current_device", lambda: 0)
        monkeypatch.setattr(
            "torch.cuda.get_device_properties",
            lambda _device: type("_Props", (), {"total_memory": 4 * 2**30})(),
        )

        class FakeSummaryNet:
            item_chunk_size = None
            gradient_checkpointing = False
            _config = {"item_chunk_size": None, "gradient_checkpointing": False}

        class FakeApprox:
            def __init__(self):
                self.summary_network = FakeSummaryNet()
                self.built = False
                self.seen_shapes = None
                self.adapter = lambda batch: batch

            def build(self, data_shapes):
                self.seen_shapes = data_shapes
                self.built = True

            def train_step(self, _dummy):
                return {"loss": 0.0}

        class FakeSimulator:
            def clone_with_sizes(self, n_items=None, n_persons=None):
                return self

            def sample(self, _batch_shape):
                return {
                    "inference_variables": np.zeros((2, 3, 2), dtype=np.float32),
                    "summary_variables": np.zeros((2, 5, 3), dtype=np.float32),
                    "inference_conditions": np.zeros((2, 2), dtype=np.float32),
                }

        approx = FakeApprox()
        result = utils_mod._probe_one(
            approx,
            FakeSimulator(),
            batch_size=2,
            item_chunk_size=None,
            gradient_checkpointing=False,
            n_items=(3, 4),
            n_persons=(5, 6),
        )

        assert result.ok is True
        assert approx.built is True
        assert approx.seen_shapes == {
            "inference_variables": (2, 3, 2),
            "summary_variables": (2, 5, 3),
            "inference_conditions": (2, 2),
        }

    def test_probe_one_retries_reset_then_succeeds(self, monkeypatch):
        """A single failed reset_peak_memory_stats() call should not doom
        the trial: _probe_one retries once before giving up."""
        reset_calls = []

        def flaky_reset():
            reset_calls.append(1)
            if len(reset_calls) == 1:
                raise RuntimeError("CUDA error: an illegal memory access")

        monkeypatch.setattr("torch.cuda.reset_peak_memory_stats", flaky_reset)
        monkeypatch.setattr("torch.cuda.max_memory_allocated", lambda: 0)
        monkeypatch.setattr("torch.cuda.max_memory_reserved", lambda: 0)
        monkeypatch.setattr("torch.cuda.current_device", lambda: 0)
        monkeypatch.setattr(
            "torch.cuda.get_device_properties",
            lambda _device: type("_Props", (), {"total_memory": 4 * 2**30})(),
        )

        class FakeSummaryNet:
            item_chunk_size = None
            gradient_checkpointing = False
            _config = {"item_chunk_size": None, "gradient_checkpointing": False}

        class FakeApprox:
            built = True
            summary_network = FakeSummaryNet()
            adapter = staticmethod(lambda batch: batch)

            def train_step(self, _dummy):
                return {"loss": 0.0}

        class FakeSimulator:
            def clone_with_sizes(self, n_items=None, n_persons=None):
                return self

            def sample(self, _batch_shape):
                return {
                    "inference_variables": np.zeros((2, 3, 2), dtype=np.float32),
                    "summary_variables": np.zeros((2, 5, 3), dtype=np.float32),
                    "inference_conditions": np.zeros((2, 2), dtype=np.float32),
                }

        result = utils_mod._probe_one(
            FakeApprox(),
            FakeSimulator(),
            batch_size=2,
            item_chunk_size=None,
            gradient_checkpointing=False,
            n_items=(3, 4),
            n_persons=(5, 6),
        )

        assert len(reset_calls) == 2
        assert result.ok is True

    def test_probe_one_raises_when_reset_fails_twice(self, monkeypatch):
        """A reset failure that persists after one retry is a CUDA/driver
        problem, not a memory-feasibility result -- it must raise instead
        of being reported as a ProbeResult with reason='reset'. Silently
        returning ok=False here would make callers like
        _search_chunk_size_boundary and probe_memory_config treat an
        infrastructure failure identically to a genuine OOM/infeasible
        config (narrowing the search boundary, or reporting "all
        configurations failed (OOM or unsupported)" for something that was
        never actually tested)."""

        def always_fails():
            raise RuntimeError("CUDA error: an illegal memory access")

        monkeypatch.setattr("torch.cuda.reset_peak_memory_stats", always_fails)
        monkeypatch.setattr("torch.cuda.current_device", lambda: 0)
        monkeypatch.setattr(
            "torch.cuda.get_device_properties",
            lambda _device: type("_Props", (), {"total_memory": 4 * 2**30})(),
        )

        class FakeSummaryNet:
            item_chunk_size = None
            gradient_checkpointing = False
            _config = {"item_chunk_size": None, "gradient_checkpointing": False}

        class FakeApprox:
            built = True
            summary_network = FakeSummaryNet()
            adapter = staticmethod(lambda batch: batch)

            def train_step(self, _dummy):
                return {"loss": 0.0}

        class FakeSimulator:
            def clone_with_sizes(self, n_items=None, n_persons=None):
                return self

            def sample(self, _batch_shape):
                return {
                    "inference_variables": np.zeros((2, 3, 2), dtype=np.float32),
                    "summary_variables": np.zeros((2, 5, 3), dtype=np.float32),
                    "inference_conditions": np.zeros((2, 2), dtype=np.float32),
                }

        with pytest.raises(RuntimeError, match="even after retrying"):
            utils_mod._probe_one(
                FakeApprox(),
                FakeSimulator(),
                batch_size=2,
                item_chunk_size=None,
                gradient_checkpointing=False,
                n_items=(3, 4),
                n_persons=(5, 6),
            )

    def test_probe_one_rejects_config_exceeding_physical_vram(self, monkeypatch):
        """A trial that doesn't raise OOM but reserves more than the device's
        actual VRAM must still be marked as failed.

        On Windows, the WDDM driver can silently spill CUDA allocations into
        slower shared/system memory instead of raising
        ``torch.cuda.OutOfMemoryError``, so ``train_step()`` "succeeds" while
        reserving more memory than the GPU physically has. Accepting that
        trial as ``ok=True`` would make ``probe_memory_config`` pick a config
        that pages over PCIe for the rest of training instead of falling
        through to one (e.g. gradient checkpointing) that actually fits.
        """
        device_total_bytes = 4 * 2**30  # pretend device has 4 GiB VRAM
        reserved_bytes = 5 * 2**30  # this trial reserved 5 GiB -- over budget

        monkeypatch.setattr("torch.cuda.max_memory_allocated", lambda: reserved_bytes)
        monkeypatch.setattr("torch.cuda.max_memory_reserved", lambda: reserved_bytes)
        monkeypatch.setattr("torch.cuda.reset_peak_memory_stats", lambda: None)
        monkeypatch.setattr("torch.cuda.current_device", lambda: 0)
        monkeypatch.setattr(
            "torch.cuda.get_device_properties",
            lambda _device: type("_Props", (), {"total_memory": device_total_bytes})(),
        )

        class FakeSummaryNet:
            item_chunk_size = None
            gradient_checkpointing = False
            _config = {"item_chunk_size": None, "gradient_checkpointing": False}

        class FakeApprox:
            def __init__(self):
                self.summary_network = FakeSummaryNet()
                self.built = True
                self.adapter = lambda batch: batch

            def train_step(self, _dummy):
                return {"loss": 0.0}

        class FakeSimulator:
            def clone_with_sizes(self, n_items=None, n_persons=None):
                return self

            def sample(self, _batch_shape):
                return {
                    "inference_variables": np.zeros((2, 3, 2), dtype=np.float32),
                    "summary_variables": np.zeros((2, 5, 3), dtype=np.float32),
                    "inference_conditions": np.zeros((2, 2), dtype=np.float32),
                }

        result = utils_mod._probe_one(
            FakeApprox(),
            FakeSimulator(),
            batch_size=2,
            item_chunk_size=None,
            gradient_checkpointing=False,
            n_items=(3, 4),
            n_persons=(5, 6),
        )

        assert result.ok is False
        # Peak memory is still recorded on a failed trial (it was measured
        # successfully -- only the acceptance check failed), so the console
        # output can show how close the config came instead of "n/a".
        assert result.peak_reserved_mib is not None
        assert result.reason == "spill"

    def test_probe_one_rejects_config_below_safety_margin(self, monkeypatch):
        """A trial that fits under physical VRAM but leaves too little headroom
        must still be marked as failed.

        A single probed train_step may not represent peak memory over
        variable-shape training.
        """
        device_total_bytes = 4 * 2**30  # pretend device has 4 GiB VRAM
        # Fits under the 4 GiB device total, but uses 95% of it -- above the
        # default 90% safety_margin.
        reserved_bytes = int(0.95 * device_total_bytes)

        monkeypatch.setattr("torch.cuda.max_memory_allocated", lambda: reserved_bytes)
        monkeypatch.setattr("torch.cuda.max_memory_reserved", lambda: reserved_bytes)
        monkeypatch.setattr("torch.cuda.reset_peak_memory_stats", lambda: None)
        monkeypatch.setattr("torch.cuda.current_device", lambda: 0)
        monkeypatch.setattr(
            "torch.cuda.get_device_properties",
            lambda _device: type("_Props", (), {"total_memory": device_total_bytes})(),
        )
        # Nothing else holds VRAM here, so the usable budget equals physical
        # VRAM and this test isolates the safety_margin check itself (see
        # test_probe_one_margin_uses_free_not_total_vram for the case where
        # the two differ).
        monkeypatch.setattr(
            "torch.cuda.mem_get_info", lambda: (device_total_bytes, device_total_bytes)
        )
        monkeypatch.setattr("torch.cuda.memory_reserved", lambda: 0)

        class FakeSummaryNet:
            item_chunk_size = None
            gradient_checkpointing = False
            _config = {"item_chunk_size": None, "gradient_checkpointing": False}

        class FakeApprox:
            def __init__(self):
                self.summary_network = FakeSummaryNet()
                self.built = True
                self.adapter = lambda batch: batch

            def train_step(self, _dummy):
                return {"loss": 0.0}

        class FakeSimulator:
            def clone_with_sizes(self, n_items=None, n_persons=None):
                return self

            def sample(self, _batch_shape):
                return {
                    "inference_variables": np.zeros((2, 3, 2), dtype=np.float32),
                    "summary_variables": np.zeros((2, 5, 3), dtype=np.float32),
                    "inference_conditions": np.zeros((2, 2), dtype=np.float32),
                }

        result = utils_mod._probe_one(
            FakeApprox(),
            FakeSimulator(),
            batch_size=2,
            item_chunk_size=None,
            gradient_checkpointing=False,
            n_items=(3, 4),
            n_persons=(5, 6),
            safety_margin=0.9,
        )

        assert result.ok is False
        assert result.peak_reserved_mib is not None
        assert result.reason == "margin"

    def test_probe_one_margin_uses_free_not_total_vram(self, monkeypatch):
        """safety_margin must be enforced against VRAM actually obtainable by
        this process, not the card's physical VRAM.

        Example: on a card where other processes hold part of the VRAM, a
        trial can look comfortable against *physical* VRAM while consuming most of
        what is actually obtainable, spending the safety margin before the
        first trial runs.
        """
        gib = 2**30
        device_total_bytes = 6 * gib
        free_bytes = 5 * gib  # ~1 GiB held by other processes
        reserved_bytes = int(4.8 * gib)  # 80% of total, but 96% of free

        monkeypatch.setattr("torch.cuda.max_memory_allocated", lambda: reserved_bytes)
        monkeypatch.setattr("torch.cuda.max_memory_reserved", lambda: reserved_bytes)
        monkeypatch.setattr("torch.cuda.reset_peak_memory_stats", lambda: None)
        monkeypatch.setattr("torch.cuda.current_device", lambda: 0)
        monkeypatch.setattr(
            "torch.cuda.get_device_properties",
            lambda _device: type("_Props", (), {"total_memory": device_total_bytes})(),
        )
        monkeypatch.setattr(
            "torch.cuda.mem_get_info", lambda: (free_bytes, device_total_bytes)
        )
        monkeypatch.setattr("torch.cuda.memory_reserved", lambda: 0)

        class FakeSummaryNet:
            item_chunk_size = None
            gradient_checkpointing = False
            _config = {"item_chunk_size": None, "gradient_checkpointing": False}

        class FakeApprox:
            def __init__(self):
                self.summary_network = FakeSummaryNet()
                self.built = True
                self.adapter = lambda batch: batch

            def train_step(self, _dummy):
                return {"loss": 0.0}

        class FakeSimulator:
            def clone_with_sizes(self, n_items=None, n_persons=None):
                return self

            def sample(self, _batch_shape):
                return {
                    "inference_variables": np.zeros((2, 3, 2), dtype=np.float32),
                    "summary_variables": np.zeros((2, 5, 3), dtype=np.float32),
                    "inference_conditions": np.zeros((2, 2), dtype=np.float32),
                }

        result = utils_mod._probe_one(
            FakeApprox(),
            FakeSimulator(),
            batch_size=2,
            item_chunk_size=None,
            gradient_checkpointing=False,
            n_items=(3, 4),
            n_persons=(5, 6),
            safety_margin=0.9,
        )

        # 4.8 GiB is 80% of the 6 GiB card -- under a 90% margin on physical
        # VRAM, so a check against physical VRAM would wrongly accept this
        # config. Against the 5 GiB actually obtainable
        # it is 96%, so it must be rejected.
        assert result.ok is False
        assert result.reason == "margin"
        assert result.device_budget_bytes == free_bytes
        assert result.device_total_bytes == device_total_bytes
        # The reported percentage uses the usable budget as denominator.
        assert result.peak_reserved_pct == pytest.approx(96.0, abs=0.5)

    def test_probe_one_budget_falls_back_to_total_when_query_fails(self, monkeypatch):
        """If the free-memory query is unavailable, the margin check must fall
        back to physical VRAM rather than failing the trial outright."""
        gib = 2**30
        device_total_bytes = 4 * gib
        reserved_bytes = int(0.5 * device_total_bytes)  # comfortably inside

        def _boom():
            raise RuntimeError("mem_get_info unavailable")

        monkeypatch.setattr("torch.cuda.max_memory_allocated", lambda: reserved_bytes)
        monkeypatch.setattr("torch.cuda.max_memory_reserved", lambda: reserved_bytes)
        monkeypatch.setattr("torch.cuda.reset_peak_memory_stats", lambda: None)
        monkeypatch.setattr("torch.cuda.current_device", lambda: 0)
        monkeypatch.setattr(
            "torch.cuda.get_device_properties",
            lambda _device: type("_Props", (), {"total_memory": device_total_bytes})(),
        )
        monkeypatch.setattr("torch.cuda.mem_get_info", _boom)

        class FakeSummaryNet:
            item_chunk_size = None
            gradient_checkpointing = False
            _config = {"item_chunk_size": None, "gradient_checkpointing": False}

        class FakeApprox:
            def __init__(self):
                self.summary_network = FakeSummaryNet()
                self.built = True
                self.adapter = lambda batch: batch

            def train_step(self, _dummy):
                return {"loss": 0.0}

        class FakeSimulator:
            def clone_with_sizes(self, n_items=None, n_persons=None):
                return self

            def sample(self, _batch_shape):
                return {
                    "inference_variables": np.zeros((2, 3, 2), dtype=np.float32),
                    "summary_variables": np.zeros((2, 5, 3), dtype=np.float32),
                    "inference_conditions": np.zeros((2, 2), dtype=np.float32),
                }

        result = utils_mod._probe_one(
            FakeApprox(),
            FakeSimulator(),
            batch_size=2,
            item_chunk_size=None,
            gradient_checkpointing=False,
            n_items=(3, 4),
            n_persons=(5, 6),
            safety_margin=0.9,
        )

        assert result.ok is True
        assert result.device_budget_bytes is None
        # Falls back to physical VRAM for the reported percentage too.
        assert result.peak_reserved_pct == pytest.approx(50.0, abs=0.5)

    def test_probe_one_reason_is_oom_on_cuda_oom(self, monkeypatch):
        """A genuine CUDA OOM during train_step must be tagged reason='oom',
        distinct from the 'spill'/'margin'/'ckpt' outcome codes."""
        monkeypatch.setattr("torch.cuda.reset_peak_memory_stats", lambda: None)
        monkeypatch.setattr("torch.cuda.current_device", lambda: 0)
        monkeypatch.setattr(
            "torch.cuda.get_device_properties",
            lambda _device: type("_Props", (), {"total_memory": 4 * 2**30})(),
        )

        class FakeSummaryNet:
            item_chunk_size = None
            gradient_checkpointing = False
            _config = {"item_chunk_size": None, "gradient_checkpointing": False}

        class FakeApprox:
            def __init__(self):
                self.summary_network = FakeSummaryNet()
                self.built = True
                self.adapter = lambda batch: batch

            def train_step(self, _dummy):
                raise RuntimeError("CUDA out of memory. Tried to allocate...")

        class FakeSimulator:
            def clone_with_sizes(self, n_items=None, n_persons=None):
                return self

            def sample(self, _batch_shape):
                return {
                    "inference_variables": np.zeros((2, 3, 2), dtype=np.float32),
                    "summary_variables": np.zeros((2, 5, 3), dtype=np.float32),
                    "inference_conditions": np.zeros((2, 2), dtype=np.float32),
                }

        result = utils_mod._probe_one(
            FakeApprox(),
            FakeSimulator(),
            batch_size=2,
            item_chunk_size=None,
            gradient_checkpointing=False,
            n_items=(3, 4),
            n_persons=(5, 6),
        )

        assert result.ok is False
        assert result.reason == "oom"

    def test_probe_one_reason_is_ckpt_on_checkpoint_error(self, monkeypatch):
        """Gradient-checkpointing incompatible with the layer config must be
        tagged reason='ckpt', not conflated with a generic OOM."""
        from torch.utils.checkpoint import CheckpointError

        monkeypatch.setattr("torch.cuda.reset_peak_memory_stats", lambda: None)
        monkeypatch.setattr("torch.cuda.current_device", lambda: 0)
        monkeypatch.setattr(
            "torch.cuda.get_device_properties",
            lambda _device: type("_Props", (), {"total_memory": 4 * 2**30})(),
        )

        class FakeSummaryNet:
            item_chunk_size = None
            gradient_checkpointing = True
            _config = {"item_chunk_size": None, "gradient_checkpointing": True}

        class FakeApprox:
            def __init__(self):
                self.summary_network = FakeSummaryNet()
                self.built = True
                self.adapter = lambda batch: batch

            def train_step(self, _dummy):
                raise CheckpointError("checkpointing is not compatible here")

        class FakeSimulator:
            def clone_with_sizes(self, n_items=None, n_persons=None):
                return self

            def sample(self, _batch_shape):
                return {
                    "inference_variables": np.zeros((2, 3, 2), dtype=np.float32),
                    "summary_variables": np.zeros((2, 5, 3), dtype=np.float32),
                    "inference_conditions": np.zeros((2, 2), dtype=np.float32),
                }

        result = utils_mod._probe_one(
            FakeApprox(),
            FakeSimulator(),
            batch_size=2,
            item_chunk_size=None,
            gradient_checkpointing=True,
            n_items=(3, 4),
            n_persons=(5, 6),
        )

        assert result.ok is False
        assert result.reason == "ckpt"

    def test_probe_uses_exact_max_ranges(self, monkeypatch):
        monkeypatch.setattr("torch.cuda.is_available", lambda: True)
        monkeypatch.setattr(
            utils_mod,
            "_build_probe_approximator",
            lambda approx: self._make_fake_approx(),
        )
        seen = []

        def fake_probe(*args, **kwargs):
            seen.append((args[5], args[6]))
            return ProbeResult(
                item_chunk_size=None,
                gradient_checkpointing=False,
                ok=True,
                peak_allocated_bytes=2**20,
                peak_reserved_bytes=2**20,
            )

        monkeypatch.setattr(utils_mod, "_probe_one", fake_probe)

        report = probe_memory_config(
            self._make_fake_approx(),
            lambda **kwargs: None,
            batch_size=8,
            n_items=(10, 20),
            n_persons=(30, 40),
            search=ChunkSearchOptions(
                candidate_item_chunk_sizes=[None], checkpoint_options=[False]
            ),
            verbose=False,
        )

        assert report.best is not None
        assert seen == [((20, 20), (40, 40))]

    @pytest.mark.parametrize("backoff_factor", [0.0, -0.5, 1.5, 2.0])
    def test_rejects_invalid_backoff_factor(self, backoff_factor):
        """backoff_factor is a float multiplier and must lie in (0, 1].
        A caller passing a divisor-style value (e.g. 2) must get a clear
        ValueError, not silently have the chunk-size grid grow instead of
        shrink."""
        with pytest.raises(ValueError, match="must be a fraction in"):
            probe_memory_config(
                self._make_fake_approx(),
                lambda **kwargs: None,
                batch_size=8,
                n_items=10,
                n_persons=100,
                search=ChunkSearchOptions(backoff_factor=backoff_factor),
                verbose=False,
            )

    def test_accepts_boundary_backoff_factor_one(self, monkeypatch):
        """backoff_factor=1.0 is the documented no-op upper boundary and
        must be accepted (not rejected as > 1 due to a float rounding
        slip)."""
        monkeypatch.setattr("torch.cuda.is_available", lambda: True)
        monkeypatch.setattr(
            utils_mod,
            "_build_probe_approximator",
            lambda approx: self._make_fake_approx(),
        )
        monkeypatch.setattr(
            utils_mod,
            "_probe_one",
            lambda *args, **kwargs: ProbeResult(
                item_chunk_size=None,
                gradient_checkpointing=False,
                ok=True,
                peak_allocated_bytes=2**20,
                peak_reserved_bytes=2**20,
            ),
        )

        report = probe_memory_config(
            self._make_fake_approx(),
            lambda **kwargs: None,
            batch_size=8,
            n_items=10,
            n_persons=100,
            search=ChunkSearchOptions(
                backoff_factor=1.0, candidate_item_chunk_sizes=[None]
            ),
            verbose=False,
        )
        assert report.best is not None


class _FakeHistory:
    history = {"loss": [1.0]}


class TestFitWithOomRetry:
    @pytest.mark.parametrize("backoff_factor", [0.0, -0.5, 1.5, 2.0])
    def test_rejects_invalid_backoff_factor(self, backoff_factor):
        """Mirrors probe_memory_config's ChunkSearchOptions.backoff_factor
        validation: it is a float multiplier and must lie in (0, 1]. A
        caller passing 2 must get a clear ValueError, not have batch/chunk
        sizes silently grow instead of shrink."""

        class FakeApprox:
            def fit(self, **kwargs):
                return _FakeHistory()

        with pytest.raises(ValueError, match="must be a fraction in"):
            fit_with_oom_retry(
                FakeApprox(),
                None,
                start_batch_size=32,
                backoff_factor=backoff_factor,
                fit_kwargs=dict(epochs=1),
            )

    def test_success_first_try(self):
        """fit succeeds immediately -> attempts == 1, no chunking."""

        class FakeApprox:
            def fit(self, **kwargs):
                return _FakeHistory()

        result = fit_with_oom_retry(
            FakeApprox(),
            None,
            start_batch_size=64,
            fit_kwargs=dict(epochs=1),
        )
        assert result.batch_size == 64
        assert result.item_chunk_size is None
        assert result.attempts == 1
        assert isinstance(result.config, TrainingMemoryConfig)

    def test_injects_periodic_cuda_cache_release_by_default(self):
        """fit_with_oom_retry should inject the cache-release callback on
        the torch backend, alongside any user-supplied callbacks."""
        seen_kwargs = {}

        class FakeApprox:
            def fit(self, **kwargs):
                seen_kwargs.update(kwargs)
                return _FakeHistory()

        user_callback = object()
        fit_with_oom_retry(
            FakeApprox(),
            None,
            start_batch_size=64,
            fit_kwargs=dict(epochs=1, callbacks=[user_callback]),
        )

        callbacks = seen_kwargs["callbacks"]
        assert user_callback in callbacks
        assert any(type(cb).__name__ == "_PeriodicCudaCacheRelease" for cb in callbacks)

    def test_release_cuda_cache_every_step_false_skips_injection(self):
        """Disabling release_cuda_cache_every_step should leave callbacks untouched."""
        seen_kwargs = {}

        class FakeApprox:
            def fit(self, **kwargs):
                seen_kwargs.update(kwargs)
                return _FakeHistory()

        fit_with_oom_retry(
            FakeApprox(),
            None,
            start_batch_size=64,
            release_cuda_cache_every_step=False,
            fit_kwargs=dict(epochs=1),
        )

        assert "callbacks" not in seen_kwargs

    def test_oom_then_success_no_summary_net(self):
        """Persistent retry succeeds at same batch_size before any halving occurs."""
        call_count = 0

        class FakeApprox:
            def fit(self, **kwargs):
                nonlocal call_count
                call_count += 1
                if call_count == 1:
                    raise RuntimeError("CUDA out of memory")
                return _FakeHistory()

        result = fit_with_oom_retry(
            FakeApprox(),
            None,
            start_batch_size=32,
            fit_kwargs=dict(epochs=1),
        )
        # Second persistent try succeeds at the same batch_size — no halving.
        assert result.batch_size == 32
        assert result.item_chunk_size is None
        assert result.attempts == 2

    def test_oom_tries_chunks_before_halving_batch(self):
        """With summary_net, OOM tries smaller chunks before halving batch."""
        attempts = []

        class FakeSummaryNet:
            item_chunk_size = None
            gradient_checkpointing = False
            _config = {"item_chunk_size": None, "gradient_checkpointing": False}

        class FakeApprox:
            summary_network = FakeSummaryNet()

            def fit(self, *, batch_size, **kwargs):
                attempts.append((batch_size, self.summary_network.item_chunk_size))
                # Fail until item_chunk_size is small enough
                if self.summary_network.item_chunk_size is None:
                    raise RuntimeError("CUDA out of memory")
                return _FakeHistory()

        result = fit_with_oom_retry(
            FakeApprox(),
            None,
            start_batch_size=32,
            fit_kwargs=dict(epochs=1),
        )
        # batch_size should NOT have been halved — chunking resolved it
        assert result.batch_size == 32
        # item_chunk_size=None is tried n_persistent_tries=2 times, then steps to 32
        assert attempts[:2] == [(32, None)] * 2
        assert attempts[2] == (32, 32)
        assert result.item_chunk_size == 32

    def test_item_chunk_ladder_terminates_with_backoff_factor_one(self):
        """backoff_factor=1 must not stall the item_chunk_size ladder forever.

        ``cs * 1 == cs``, so a naive ``cs = int(cs * backoff_factor)`` step
        never shrinks ``cs`` and the ``while cs >= 8`` loop building the
        ladder would never terminate. The ladder builder guards against this
        by falling back to ``cs - 1`` whenever the multiplied step doesn't
        actually shrink ``cs``. This test would hang if that guard
        regressed.
        """
        attempts = []

        class FakeSummaryNet:
            item_chunk_size = None
            gradient_checkpointing = False
            _config = {"item_chunk_size": None, "gradient_checkpointing": False}

        class FakeApprox:
            summary_network = FakeSummaryNet()

            def fit(self, *, batch_size, **kwargs):
                attempts.append((batch_size, self.summary_network.item_chunk_size))
                if self.summary_network.item_chunk_size != 8:
                    raise RuntimeError("CUDA out of memory")
                return _FakeHistory()

        result = fit_with_oom_retry(
            FakeApprox(),
            None,
            start_batch_size=10,
            backoff_factor=1,
            n_persistent_tries=1,
            fit_kwargs=dict(epochs=1),
        )
        # Ladder decrements by 1 per rung (10 -> 9 -> 8) instead of stalling.
        assert attempts == [(10, None), (10, 10), (10, 9), (10, 8)]
        assert result.batch_size == 10
        assert result.item_chunk_size == 8

    def test_oom_halves_batch_after_all_chunks_exhausted(self):
        """Batch is halved only once all chunk sizes at a given batch_size fail."""
        seen = []

        class FakeSummaryNet:
            item_chunk_size = None
            gradient_checkpointing = False
            _config = {"item_chunk_size": None, "gradient_checkpointing": False}

        class FakeApprox:
            summary_network = FakeSummaryNet()

            def fit(self, *, batch_size, **kwargs):
                seen.append((batch_size, self.summary_network.item_chunk_size))
                # All attempts at batch=32 fail; succeed at batch=16
                if batch_size == 32:
                    raise RuntimeError("CUDA out of memory")
                return _FakeHistory()

        result = fit_with_oom_retry(
            FakeApprox(),
            None,
            start_batch_size=32,
            min_batch_size=8,
            fit_kwargs=dict(epochs=1),
        )
        assert result.batch_size == 16
        # All attempts at batch=32 should have occurred before moving to batch=16
        batch32_attempts = [s for s in seen if s[0] == 32]
        batch16_attempts = [s for s in seen if s[0] == 16]
        assert len(batch32_attempts) > 0
        assert len(batch16_attempts) > 0
        assert all(s[0] == 32 for s in seen[: len(batch32_attempts)])

    def test_oom_retry_halves_batch_when_summary_opts_out_of_chunking(self):
        """A summary with item_chunk_size but no chunk support skips chunk ladder."""
        seen = []

        class FakeSummaryNet:
            supports_item_chunking = False
            item_chunk_size = None
            gradient_checkpointing = False
            _config = {"item_chunk_size": None, "gradient_checkpointing": False}

        class FakeApprox:
            summary_network = FakeSummaryNet()

            def fit(self, *, batch_size, **kwargs):
                seen.append((batch_size, self.summary_network.item_chunk_size))
                if batch_size == 32:
                    raise RuntimeError("CUDA out of memory")
                return _FakeHistory()

        result = fit_with_oom_retry(
            FakeApprox(),
            None,
            start_batch_size=32,
            min_batch_size=16,
            fit_kwargs=dict(epochs=1),
        )

        assert result.batch_size == 16
        assert result.item_chunk_size is None
        assert seen[:2] == [(32, None)] * 2
        assert seen[2] == (16, None)

    def test_repeated_oom_raises(self):
        """OOM all the way down -> RuntimeError."""

        class FakeApprox:
            def fit(self, **kwargs):
                raise RuntimeError("CUDA out of memory")

        with pytest.raises(RuntimeError, match="OOM"):
            fit_with_oom_retry(
                FakeApprox(),
                None,
                start_batch_size=16,
                min_batch_size=8,
                fit_kwargs=dict(epochs=1),
            )

    def test_non_oom_passthrough(self):
        """Non-OOM exceptions propagate immediately."""

        class FakeApprox:
            def fit(self, **kwargs):
                raise ValueError("bad value")

        with pytest.raises(ValueError, match="bad value"):
            fit_with_oom_retry(
                FakeApprox(),
                None,
                start_batch_size=32,
                fit_kwargs=dict(epochs=1),
            )

    def test_backoff_factor(self):
        """Custom backoff_factor scales batch size correctly (no summary_net)."""
        call_batch_sizes = []

        class FakeApprox:
            def fit(self, *, batch_size, **kwargs):
                call_batch_sizes.append(batch_size)
                if batch_size > 8:
                    raise RuntimeError("CUDA out of memory")
                return _FakeHistory()

        result = fit_with_oom_retry(
            FakeApprox(),
            None,
            start_batch_size=64,
            min_batch_size=4,
            backoff_factor=0.25,
            fit_kwargs=dict(epochs=1),
        )
        # 64 -> 16 -> 4 (4 >= min_batch_size=4, succeeds)
        assert result.batch_size in call_batch_sizes

    def test_backoff_factor_one_still_makes_progress(self):
        """backoff_factor=1.0 must not stall the batch-size step-down forever.

        ``batch_size * 1.0 == batch_size``, so a naive
        ``batch_size = int(batch_size * backoff_factor)`` step never shrinks
        batch_size and the outer retry loop would repeat the same failing
        batch_size forever instead of ever reaching min_batch_size and
        raising. The step-down must fall back to a strict decrement, the
        same guard already used for the item/person chunk-size ladders.
        """
        call_batch_sizes = []

        class FakeApprox:
            def fit(self, *, batch_size, **kwargs):
                call_batch_sizes.append(batch_size)
                raise RuntimeError("CUDA out of memory")

        with pytest.raises(RuntimeError, match="OOM"):
            fit_with_oom_retry(
                FakeApprox(),
                None,
                start_batch_size=6,
                min_batch_size=4,
                backoff_factor=1.0,
                n_persistent_tries=1,
                fit_kwargs=dict(epochs=1),
            )
        # Strictly decreasing (6 -> 5 -> 4), not stuck at 6 forever.
        assert call_batch_sizes == [6, 5, 4]

    def test_backoff_factor_fractional_reciprocal_does_not_compensate(self):
        """A backoff_factor whose reciprocal is not an integer must not
        silently shrink the effective batch size while claiming to
        compensate via gradient_accumulation_steps.

        With gradient_accumulation_steps=1 and backoff_factor=0.75,
        ``int(1 / 0.75) == 1`` -- the proposed accumulation is unchanged,
        which trivially divides any num_batches, but the batch size still
        shrinks by 25%. Compensation must be rejected in this case (falling
        back to a plain, uncompensated batch cut) rather than silently
        breaking the "effective batch size held constant" guarantee.
        """

        class FakeApprox:
            gradient_accumulation_steps = 1

            def fit(self, *, batch_size, **kwargs):
                if batch_size > 6:
                    raise RuntimeError("CUDA out of memory")
                return _FakeHistory()

        approx = FakeApprox()
        result = fit_with_oom_retry(
            approx,
            None,
            start_batch_size=8,
            min_batch_size=1,
            backoff_factor=0.75,
            fit_kwargs=dict(epochs=1, num_batches=4),
        )
        # 8 -> 6 (int(8 * 0.75)), uncompensated: accumulation stays at 1.
        assert result.batch_size == 6
        assert approx.gradient_accumulation_steps == 1

    def test_persistent_retry_same_config_before_stepdown(self):
        """All persistent tries exhaust at one config before any step-down.

        Success on the last try stays at the same batch size.
        """
        call_count = 0
        n = 2  # default n_persistent_tries

        class FakeApprox:
            def fit(self, **kwargs):
                nonlocal call_count
                call_count += 1
                # OOM for the first n-1 tries, succeed on the nth try
                if call_count < n:
                    raise RuntimeError("CUDA out of memory")
                return _FakeHistory()

        result = fit_with_oom_retry(
            FakeApprox(),
            None,
            start_batch_size=32,
            fit_kwargs=dict(epochs=1),
        )
        assert result.batch_size == 32  # no step-down happened
        assert result.item_chunk_size is None
        assert result.attempts == n  # exactly n_persistent_tries

    def test_persistent_retry_exhausted_then_steps_down(self):
        """After n_persistent_tries OOMs at one config, the function steps down (halves
        batch_size when there is no summary_net)."""
        seen_batch_sizes = []
        n = 2  # default n_persistent_tries

        class FakeApprox:
            def fit(self, *, batch_size, **kwargs):
                seen_batch_sizes.append(batch_size)
                if batch_size == 32:
                    raise RuntimeError("CUDA out of memory")
                return _FakeHistory()

        result = fit_with_oom_retry(
            FakeApprox(),
            None,
            start_batch_size=32,
            fit_kwargs=dict(epochs=1),
        )
        assert result.batch_size == 16
        # All n_persistent_tries attempts at batch=32 happen before the step-down
        assert seen_batch_sizes[:n] == [32] * n
        assert seen_batch_sizes[n] == 16
        assert result.attempts == n + 1

    def test_oom_tries_person_chunks_before_halving_batch(self):
        """With a person-chunking summary_net and no start_person_chunk_size
        given (the default, unseeded case), OOM tries dense first, then
        smaller person chunks, before halving batch (mirrors the
        item_chunk_size ladder test above)."""
        attempts = []

        class FakeSummaryNet:
            supports_item_chunking = False
            supports_person_chunking = True
            item_chunk_size = None
            person_chunk_size = None
            gradient_checkpointing = False
            _config = {
                "item_chunk_size": None,
                "gradient_checkpointing": False,
                "person_chunk_size": None,
            }

        class FakeApprox:
            summary_network = FakeSummaryNet()

            def fit(self, *, batch_size, **kwargs):
                attempts.append((batch_size, self.summary_network.person_chunk_size))
                if self.summary_network.person_chunk_size is None:
                    raise RuntimeError("CUDA out of memory")
                return _FakeHistory()

        result = fit_with_oom_retry(
            FakeApprox(),
            None,
            start_batch_size=32,
            fit_kwargs=dict(epochs=1),
        )
        # batch_size should NOT have been halved — person chunking resolved it
        assert result.batch_size == 32
        assert attempts[:2] == [(32, None)] * 2
        # First non-None rung of the default (unseeded) ladder is the
        # internal default starting point, 8192.
        assert attempts[2] == (32, 8192)
        assert result.person_chunk_size == 8192
        assert result.item_chunk_size is None

    def test_unseeded_person_chunk_ladder_clamps_to_simulator_n_persons(self):
        """The unseeded ladder's internal default starting rung (8192) is a
        functional no-op when it's >= the simulator's actual max n_persons
        (chunk_size >= n_persons chunks nothing, same as the dense rung
        already tried) -- it should clamp below that ceiling instead of
        wasting a real OOM retry re-proving the dense failure."""
        attempts = []

        class FakeSummaryNet:
            supports_item_chunking = False
            supports_person_chunking = True
            item_chunk_size = None
            person_chunk_size = None
            gradient_checkpointing = False
            _config = {
                "item_chunk_size": None,
                "gradient_checkpointing": False,
                "person_chunk_size": None,
            }

        class FakeApprox:
            summary_network = FakeSummaryNet()

            def fit(self, *, batch_size, **kwargs):
                attempts.append((batch_size, self.summary_network.person_chunk_size))
                if self.summary_network.person_chunk_size is None:
                    raise RuntimeError("CUDA out of memory")
                return _FakeHistory()

        class FakeSimulator:
            n_persons = (5, 500)

        result = fit_with_oom_retry(
            FakeApprox(),
            FakeSimulator(),
            start_batch_size=32,
            fit_kwargs=dict(epochs=1),
        )
        assert attempts[:2] == [(32, None)] * 2
        # First non-None rung must be < 500 (the simulator's max n_persons),
        # not the unclamped internal default of 8192.
        assert attempts[2] == (32, 499)
        assert result.person_chunk_size == 499

    def test_explicit_start_person_chunk_size_skips_none_rung(self):
        """An explicit start_person_chunk_size is treated as already
        validated (e.g. via probe_memory_config): the very first attempt
        should already use it, not None -- retrying dense first would only
        re-derive, via a real (possibly slow) training attempt, a config
        already known to be infeasible."""
        attempts = []

        class FakeSummaryNet:
            supports_item_chunking = False
            supports_person_chunking = True
            item_chunk_size = None
            person_chunk_size = None
            gradient_checkpointing = False
            _config = {
                "item_chunk_size": None,
                "gradient_checkpointing": False,
                "person_chunk_size": None,
            }

        class FakeApprox:
            summary_network = FakeSummaryNet()

            def fit(self, *, batch_size, **kwargs):
                attempts.append((batch_size, self.summary_network.person_chunk_size))
                return _FakeHistory()

        result = fit_with_oom_retry(
            FakeApprox(),
            None,
            start_batch_size=32,
            start_person_chunk_size=64,
            fit_kwargs=dict(epochs=1),
        )
        assert attempts == [(32, 64)]
        assert result.person_chunk_size == 64
        assert result.attempts == 1

    def test_seeded_person_chunk_size_backs_off_gently_not_halving(self):
        """A seeded (explicit) start_person_chunk_size should step down by
        sqrt(backoff_factor) on OOM, not the full backoff_factor -- a
        gentler descent than item_chunk_size/batch_size's halving, since a
        real OOM this close to an already-validated value likely means
        "just barely not enough," not license to jump to a much smaller,
        slower chunk size."""
        attempts = []

        class FakeSummaryNet:
            supports_item_chunking = False
            supports_person_chunking = True
            item_chunk_size = None
            person_chunk_size = None
            gradient_checkpointing = False
            _config = {
                "item_chunk_size": None,
                "gradient_checkpointing": False,
                "person_chunk_size": None,
            }

        class FakeApprox:
            summary_network = FakeSummaryNet()

            def fit(self, *, batch_size, **kwargs):
                pcs = self.summary_network.person_chunk_size
                attempts.append(pcs)
                # Only the second rung (110) succeeds.
                if pcs == 110:
                    return _FakeHistory()
                raise RuntimeError("CUDA out of memory")

        result = fit_with_oom_retry(
            FakeApprox(),
            None,
            start_batch_size=32,
            start_person_chunk_size=156,
            n_persistent_tries=1,
            fit_kwargs=dict(epochs=1),
        )
        # 156 -> 110 (int(156 * sqrt(0.5)) -- gentler than 156 -> 78 via a
        # full halving with the default backoff_factor=0.5).
        assert attempts == [156, 110]
        assert result.person_chunk_size == 110

    def test_unseeded_person_chunk_ladder_terminates_with_backoff_factor_one(self):
        """backoff_factor=1 must not stall the *unseeded* person_chunk_size
        ladder forever, mirroring the item-chunk-ladder and batch-size
        regression tests above.

        Unlike the seeded case
        (test_seeded_person_chunk_size_backs_off_gently_not_halving), no
        start_person_chunk_size is given here, so the ladder is built
        from the internal default starting rung (8192) using the raw
        backoff_factor (not sqrt(backoff_factor)). ``pcs * 1 == pcs``, so a
        naive step never shrinks pcs and the ``while pcs >= 8:`` loop that
        builds the ladder would never terminate -- this test would hang if
        the ``next_pcs >= pcs: next_pcs = pcs - 1`` guard regressed.
        """
        attempts = []

        class FakeSummaryNet:
            supports_item_chunking = False
            supports_person_chunking = True
            item_chunk_size = None
            person_chunk_size = None
            gradient_checkpointing = False
            _config = {
                "item_chunk_size": None,
                "gradient_checkpointing": False,
                "person_chunk_size": None,
            }

        class FakeApprox:
            summary_network = FakeSummaryNet()

            def fit(self, *, batch_size, **kwargs):
                pcs = self.summary_network.person_chunk_size
                attempts.append(pcs)
                if pcs == 8190:
                    return _FakeHistory()
                raise RuntimeError("CUDA out of memory")

        result = fit_with_oom_retry(
            FakeApprox(),
            None,
            start_batch_size=32,
            backoff_factor=1.0,
            n_persistent_tries=1,
            fit_kwargs=dict(epochs=1),
        )
        # Dense (None) first, then the ladder decrements by 1 per rung
        # (8192 -> 8191 -> 8190) instead of stalling at 8192 forever.
        assert attempts == [None, 8192, 8191, 8190]
        assert result.person_chunk_size == 8190

    def test_batch_halving_compensates_with_gradient_accumulation(self):
        """Halving batch_size should double gradient_accumulation_steps when
        the approximator supports it AND num_batches is known and evenly
        divisible by the proposed accumulation_steps, holding effective
        batch size constant (step 3 of the OOM backoff ladder)."""

        class FakeApprox:
            gradient_accumulation_steps = 1

            def fit(self, *, batch_size, **kwargs):
                if batch_size == 32:
                    raise RuntimeError("CUDA out of memory")
                return _FakeHistory()

        approx = FakeApprox()
        result = fit_with_oom_retry(
            approx,
            None,
            start_batch_size=32,
            min_batch_size=8,
            # num_batches=4 is evenly divisible by the proposed
            # accumulation_steps=2, so compensation is safe to apply.
            fit_kwargs=dict(epochs=1, num_batches=4),
        )
        assert result.batch_size == 16
        assert approx.gradient_accumulation_steps == 2
        assert result.config.gradient_accumulation_steps == 2

    def test_batch_halving_skips_accumulation_when_standardize_params(self):
        """standardize_params=True forbids gradient_accumulation_steps > 1
        (see EquivariantIRTApproximator.__init__), so the compensation step
        must not engage — falls back to a plain batch-size cut."""

        class FakeApprox:
            gradient_accumulation_steps = 1
            standardize_params = True

            def fit(self, *, batch_size, **kwargs):
                if batch_size == 32:
                    raise RuntimeError("CUDA out of memory")
                return _FakeHistory()

        approx = FakeApprox()
        result = fit_with_oom_retry(
            approx,
            None,
            start_batch_size=32,
            min_batch_size=8,
            fit_kwargs=dict(epochs=1, num_batches=4),
        )
        assert result.batch_size == 16
        assert approx.gradient_accumulation_steps == 1

    def test_batch_halving_skips_accumulation_when_attribute_absent(self):
        """Approximators without a gradient_accumulation_steps attribute at
        all (e.g. simplified test doubles) fall back to a plain cut too."""

        class FakeApprox:
            def fit(self, *, batch_size, **kwargs):
                if batch_size == 32:
                    raise RuntimeError("CUDA out of memory")
                return _FakeHistory()

        result = fit_with_oom_retry(
            FakeApprox(),
            None,
            start_batch_size=32,
            min_batch_size=8,
            fit_kwargs=dict(epochs=1, num_batches=4),
        )
        assert result.batch_size == 16
        assert result.config.gradient_accumulation_steps == 1

    def test_batch_halving_skips_accumulation_when_num_batches_unknown(self):
        """Without num_batches, we cannot verify the accumulation window
        completes within an epoch, so compensation must not engage (a
        gradient_accumulation_steps > num_batches would otherwise make
        EquivariantIRTApproximator's _AccumulationEpochReset callback
        silently discard every epoch's accumulated gradients)."""

        class FakeApprox:
            gradient_accumulation_steps = 1

            def fit(self, *, batch_size, **kwargs):
                if batch_size == 32:
                    raise RuntimeError("CUDA out of memory")
                return _FakeHistory()

        approx = FakeApprox()
        result = fit_with_oom_retry(
            approx,
            None,
            start_batch_size=32,
            min_batch_size=8,
            fit_kwargs=dict(epochs=1),  # no num_batches
        )
        assert result.batch_size == 16
        assert approx.gradient_accumulation_steps == 1

    def test_batch_halving_skips_accumulation_when_not_evenly_divisible(self):
        """A proposed accumulation_steps that does not evenly divide
        num_batches would discard part of every epoch's gradients, so it
        must be rejected even though num_batches is known."""

        class FakeApprox:
            gradient_accumulation_steps = 1

            def fit(self, *, batch_size, **kwargs):
                if batch_size == 32:
                    raise RuntimeError("CUDA out of memory")
                return _FakeHistory()

        approx = FakeApprox()
        result = fit_with_oom_retry(
            approx,
            None,
            start_batch_size=32,
            min_batch_size=8,
            # proposed accumulation_steps=2 does not evenly divide 3.
            fit_kwargs=dict(epochs=1, num_batches=3),
        )
        assert result.batch_size == 16
        assert approx.gradient_accumulation_steps == 1

    def test_batch_halving_skips_accumulation_when_exceeds_num_batches(self):
        """A proposed accumulation_steps larger than num_batches would never
        complete a single accumulation window within an epoch — every
        optimizer update would be silently skipped."""

        class FakeApprox:
            gradient_accumulation_steps = 1

            def fit(self, *, batch_size, **kwargs):
                if batch_size == 32:
                    raise RuntimeError("CUDA out of memory")
                return _FakeHistory()

        approx = FakeApprox()
        result = fit_with_oom_retry(
            approx,
            None,
            start_batch_size=32,
            min_batch_size=8,
            # proposed accumulation_steps=2 > num_batches=1.
            fit_kwargs=dict(epochs=1, num_batches=1),
        )
        assert result.batch_size == 16
        assert approx.gradient_accumulation_steps == 1

    def test_batch_halving_skips_accumulation_on_non_torch_backend(self, monkeypatch):
        """EquivariantIRTApproximator.train_step() raises NotImplementedError
        (not an OOM) for gradient_accumulation_steps > 1 outside the torch
        backend -- compensation must not engage there, or a recoverable OOM
        would be turned into a hard abort of the whole retry loop.

        is_oom_error is itself backend-aware (it branches on
        keras.backend.backend()), so it is stubbed out here too -- this test
        targets fit_with_oom_retry's compensation gate specifically, not
        is_oom_error's own backend-detection logic (covered separately by
        TestIsOomError)."""
        monkeypatch.setattr(utils_mod, "is_oom_error", lambda exc: True)
        monkeypatch.setattr(utils_mod.keras.backend, "backend", lambda: "tensorflow")

        class FakeApprox:
            gradient_accumulation_steps = 1

            def fit(self, *, batch_size, **kwargs):
                if batch_size == 32:
                    raise RuntimeError("CUDA out of memory")
                return _FakeHistory()

        approx = FakeApprox()
        result = fit_with_oom_retry(
            approx,
            None,
            start_batch_size=32,
            min_batch_size=8,
            fit_kwargs=dict(epochs=1, num_batches=4),
        )
        assert result.batch_size == 16
        assert approx.gradient_accumulation_steps == 1

    def test_lr_resync_accounts_for_active_gradient_accumulation_steps(self):
        """Once gradient_accumulation_steps compensation raises accumulation
        above 1, optimizer.iterations advances once per accumulation window
        (num_batches // gradient_accumulation_steps per epoch), not once per
        batch (see EquivariantIRTApproximator.train_step). The LR resync
        after a later OOM must clamp optimizer.iterations to the actual
        number of completed optimizer.apply() calls, not
        epochs_done * num_batches -- the latter would overshoot by a factor
        of gradient_accumulation_steps and prematurely decay schedules like
        CosineDecay.

        Scenario: every attempt at batch_size=32 OOMs immediately (0
        progress) until the persistent-try budget (default
        n_persistent_tries=2) is exhausted, triggering a compensated
        step-down to batch_size=16, gradient_accumulation_steps=2
        (num_batches=4 is evenly divisible). The first attempt at the
        compensated config completes exactly one epoch before OOMing again
        -- a compensated retry completing an epoch before OOMing."""

        class _FakeIterations:
            def __init__(self):
                self.value = 0

            def assign(self, v):
                self.value = v

        class _FakeOptimizer:
            def __init__(self):
                self.iterations = _FakeIterations()

        class _FakeHistoryWithLosses:
            def __init__(self, losses):
                self.history = {"loss": list(losses)}

        class FakeApprox:
            def __init__(self):
                self.gradient_accumulation_steps = 1
                self.optimizer = _FakeOptimizer()
                self.history = None
                self.call_count = 0

            def fit(self, *, batch_size, **kwargs):
                self.call_count += 1
                if batch_size == 32:
                    # No progress at all at the original batch size.
                    raise RuntimeError("CUDA out of memory")
                if batch_size == 16 and self.call_count == 3:
                    # First attempt at the compensated config: completes one
                    # epoch (with gradient_accumulation_steps=2 already
                    # active) before OOMing again.
                    self.history = _FakeHistoryWithLosses([1.0])
                    raise RuntimeError("CUDA out of memory")
                return _FakeHistory()

        approx = FakeApprox()
        result = fit_with_oom_retry(
            approx,
            None,
            start_batch_size=32,
            min_batch_size=8,
            fit_kwargs=dict(epochs=5, num_batches=4),
        )

        assert result.batch_size == 16
        assert approx.gradient_accumulation_steps == 2
        # 1 completed epoch * (num_batches=4 // gradient_accumulation_steps=2)
        # = 2 optimizer.apply() calls -- NOT epochs_done * num_batches = 4,
        # which a naive formula would assign.
        assert approx.optimizer.iterations.value == 2

    @pytest.mark.parametrize("complete_epoch_first", [True, False])
    def test_lr_resync_keeps_preexisting_optimizer_iterations(
        self, complete_epoch_first
    ):
        """A model entering the retry loop with optimizer.iterations > 0 must
        be resynced to that baseline plus the completed steps after an OOM
        that follows a completed epoch. An OOM before any epoch completed
        leaves the counter alone, because the weights and optimizer slots
        keep that attempt's partial updates."""

        class _FakeIterations:
            def __init__(self, value):
                self.value = value

            def __int__(self):
                return int(self.value)

            def assign(self, v):
                self.value = v

        class _FakeOptimizer:
            def __init__(self):
                self.iterations = _FakeIterations(100)

        class _FakeHistoryWithLosses:
            def __init__(self, losses):
                self.history = {"loss": list(losses)}

        class FakeApprox:
            def __init__(self):
                self.optimizer = _FakeOptimizer()
                self.history = None
                self.call_count = 0
                self.seen_start = []

            def fit(self, *, batch_size, **kwargs):
                self.call_count += 1
                self.seen_start.append(int(self.optimizer.iterations))
                if self.call_count == 1:
                    # Partial progress inside the failed attempt.
                    self.optimizer.iterations.assign(int(self.optimizer.iterations) + 3)
                    if complete_epoch_first:
                        self.history = _FakeHistoryWithLosses([1.0])
                    raise RuntimeError("CUDA out of memory")
                return _FakeHistory()

        approx = FakeApprox()
        fit_with_oom_retry(
            approx,
            None,
            start_batch_size=32,
            min_batch_size=8,
            fit_kwargs=dict(epochs=5, num_batches=4),
        )

        expected = 104 if complete_epoch_first else 103
        assert approx.seen_start == [100, expected]

    def test_oom_before_first_epoch_keeps_adam_step_consistent(self):
        """With a real Adam, an OOM before any completed epoch must not
        rewind the step counter that backs the retained moment slots."""
        import keras

        def make():
            var = keras.Variable(np.zeros((1,), dtype="float32"))
            opt = keras.optimizers.Adam(learning_rate=0.001)
            opt.build([var])
            return var, opt

        ref_var, ref_opt = make()
        for _ in range(4):
            ref_opt.apply([keras.ops.ones((1,))], [ref_var])

        var, opt = make()

        class FakeApprox:
            def __init__(self):
                self.optimizer = opt
                self.history = None
                self.call_count = 0

            def fit(self, *, batch_size, **kwargs):
                self.call_count += 1
                if self.call_count == 1:
                    for _ in range(3):
                        opt.apply([keras.ops.ones((1,))], [var])
                    raise RuntimeError("CUDA out of memory")
                opt.apply([keras.ops.ones((1,))], [var])
                return _FakeHistory()

        fit_with_oom_retry(
            FakeApprox(),
            None,
            start_batch_size=32,
            min_batch_size=8,
            fit_kwargs=dict(epochs=1, num_batches=4),
        )

        assert int(opt.iterations) == 4
        np.testing.assert_allclose(
            keras.ops.convert_to_numpy(var),
            keras.ops.convert_to_numpy(ref_var),
            rtol=1e-6,
        )

    def test_returned_history_covers_all_epochs_across_restarts(self):
        """The returned History must cover every epoch actually trained,
        not just the final attempt's. Example: an OOM after epoch 1 of 5
        restarts training for the remaining 4 epochs; if fit_with_oom_retry
        returned only the second attempt's History, the saved metadata
        sidecar (which reads FitResult.history.history["loss"]) would hold
        only 4 epochs of per-epoch metrics while hparams.epochs records 5.

        Scenario: epochs=3; the first attempt completes exactly epoch 1
        (loss=1.0) before OOMing; the second attempt (epochs=2, the
        remainder) succeeds with losses [0.5, 0.25]. The merged history
        must contain all three losses in training order.
        """

        class _FakeHistoryWithLosses:
            def __init__(self, losses):
                self.history = {"loss": list(losses)}

        class FakeApprox:
            def __init__(self):
                self.history = None
                self.call_count = 0

            def fit(self, *, batch_size, epochs, **kwargs):
                self.call_count += 1
                if self.call_count == 1:
                    self.history = _FakeHistoryWithLosses([1.0])
                    raise RuntimeError("CUDA out of memory")
                assert epochs == 2  # 3 total - 1 already completed
                return _FakeHistoryWithLosses([0.5, 0.25])

        result = fit_with_oom_retry(
            FakeApprox(),
            None,
            start_batch_size=32,
            fit_kwargs=dict(epochs=3),
        )

        assert result.history.history["loss"] == [1.0, 0.5, 0.25]

    def test_returned_history_on_keyboard_interrupt_covers_prior_attempts(self):
        """KeyboardInterrupt's partial-history return path must also merge
        in any earlier attempts' epochs, not just discard them.
        """

        class _FakeHistoryWithLosses:
            def __init__(self, losses):
                self.history = {"loss": list(losses)}

        class FakeApprox:
            def __init__(self):
                self.history = None
                self.call_count = 0

            def fit(self, *, batch_size, epochs, **kwargs):
                self.call_count += 1
                if self.call_count == 1:
                    self.history = _FakeHistoryWithLosses([1.0])
                    raise RuntimeError("CUDA out of memory")
                self.history = _FakeHistoryWithLosses([0.5])
                raise KeyboardInterrupt

        result = fit_with_oom_retry(
            FakeApprox(),
            None,
            start_batch_size=32,
            fit_kwargs=dict(epochs=3),
        )

        assert result.history.history["loss"] == [1.0, 0.5]


class TestConfigureLargeScaleTraining:
    def test_callable(self):
        configure_large_scale_training(
            mixed_precision=False, torch_float32_matmul_precision=None
        )

    def test_matmul_precision_import_error_downgraded_when_not_strict(
        self, monkeypatch
    ):
        """On a JAX/TensorFlow-only install (no optional torch dependency),
        `import torch` inside the matmul-precision branch raises ImportError.
        With strict=False (the non-default), this must be downgraded to a
        warning like any other setup failure, not propagate uncaught --
        narrowing the except clause to (ValueError, RuntimeError) alone would
        regress this."""
        real_import = builtins.__import__

        def fake_import(name, *args, **kwargs):
            if name == "torch":
                raise ImportError("No module named 'torch'")
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", fake_import)

        # Must not raise.
        configure_large_scale_training(
            mixed_precision=False,
            torch_float32_matmul_precision="high",
            strict=False,
        )

    def test_matmul_precision_import_error_raises_when_strict(self, monkeypatch):
        real_import = builtins.__import__

        def fake_import(name, *args, **kwargs):
            if name == "torch":
                raise ImportError("No module named 'torch'")
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", fake_import)

        with pytest.raises(RuntimeError, match="Could not set torch matmul precision"):
            configure_large_scale_training(
                mixed_precision=False,
                torch_float32_matmul_precision="high",
                strict=True,
            )
