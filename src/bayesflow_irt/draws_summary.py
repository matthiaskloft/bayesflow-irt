"""Draws summary for the autoregressive item posterior.

:class:`DrawsSummary` conditions the flow for item ``i`` on a set summary of
the items drawn before it, which turns the product of per-item posteriors into
the autoregressive item factor of Habermann et al. (2026, Sec. 2.5, Eq. 3). The
summary pairs each earlier draw with its item embedding, a simplification of the
extended observations of Sec. 2.6 (which pair parameters with raw observations
before the exchangeable encoding). It is used by
:class:`~bayesflow_irt.EquivariantIRTApproximator` with
``item_posterior="autoregressive_coupling_flow"``.

References
----------
.. [1] Habermann, D., Bulling, A., Radev, S. T., & Burkner, P.-C. (2026).
   *Amortized Bayesian inference on multilevel models of arbitrary
   structure*. arXiv. https://doi.org/10.48550/arXiv.2609.40024
"""

from __future__ import annotations

import keras
import keras.ops as ops


class DrawsSummary(keras.Layer):
    """Summary of the earlier draws for the autoregressive item factor.

    For item position ``i`` (0-based) the output is
    ``[rho(pool_{j<i} phi(x_j, h_j)), log(i + 1)]``, where ``x_j`` are the
    network-space parameters of item ``j`` and ``h_j`` its embedding from the
    matrix-equivariant summary. ``h_j`` is computed from the
    responses alone and ``x_j`` is concatenated to it afterwards. This is a
    simplification of the extended observations of Habermann et al. (2026,
    Sec. 2.6), which concatenate the conditioned parameters with each related
    raw observation *before* the exchangeable encoding; here the response
    representation does not depend on the earlier draws. The log count is
    appended as in Sec. 2.6 because pooling discards the set size. Position 0 has no
    earlier draws, so ``rho(.)`` is replaced by a learned constant ``ctx_0``.

    Pooling over the earlier draws (``pooling``):

    * ``"mean"`` -- the DeepSet mean of Habermann et al. (2026, Sec. 2.6). All
      ``I`` teacher-forced outputs come from one cumulative sum (``O(I)``): a
      DeepSet with mean pooling over a prefix is a cumulative mean.
    * ``"mean_std"`` / ``"mean_max"`` -- the mean plus the centered standard
      deviation or the elementwise max over the earlier draws. Several
      aggregators separate inputs that a single moment collapses (Corso et al.,
      2020); ``MatrixEquivariantIRTSummary`` offers the same choice for its
      person pooling.
    * ``"attention"`` -- the multihead attention block of Lee et al. (2019,
      Eqs. 6-7), ``MAB(q_i, Z) = LayerNorm(H + rFF(H))`` with
      ``H = LayerNorm(q_i + Multihead(q_i, Z, Z))`` and
      ``Z = {phi(x_j, h_j) : j < i}``. Where PMA (Eq. 11) queries with a
      learned seed, the query here is the projected embedding ``q_i`` of the
      item being drawn, so each item weighs the earlier draws by relevance to
      itself.
    * ``"pma"`` -- pooling by multihead attention with one seed (Lee et al.,
      2019, Eq. 11), ``PMA_1(Z) = MAB(S, rFF(Z))`` with a learned seed ``S``;
      ``phi`` is the row-wise feedforward ``rFF``. The weights depend on the
      earlier draws' content but not on the item being drawn, which places
      it between ``"mean"`` and ``"attention"``.

    The non-mean poolings use an explicit causal mask, ``O(I^2)`` per dataset;
    at ``I <= 50`` that is small next to the summary network.

    Parameters
    ----------
    hidden_dim : int
        Width of ``phi``, of the attention block and of the hidden layer of
        ``rho``.
    summary_dim : int
        Width of ``rho(.)`` (the output has ``summary_dim + 1`` channels
        including the log count).
    pooling : {"mean", "mean_std", "mean_max", "attention", "pma"}
        Pooling over the earlier draws.
    num_heads : int
        Attention heads (``"attention"`` and ``"pma"`` only); must divide
        ``hidden_dim``.

    References
    ----------
    Corso et al. (2020), doi:10.48550/arXiv.2004.05718; Habermann et al.
    (2026), doi:10.48550/arXiv.2609.40024, Sec. 2.6; Lee et al. (2019),
    PMLR 97, Sec. 3.1-3.2.
    """

    POOLINGS = ("mean", "mean_std", "mean_max", "attention", "pma")
    _ATTENTION_POOLINGS = ("attention", "pma")
    _STD_EPS = 1e-6

    def __init__(
        self,
        hidden_dim: int = 32,
        summary_dim: int = 32,
        pooling: str = "mean",
        num_heads: int = 4,
        **kwargs,
    ):
        super().__init__(**kwargs)
        if pooling not in self.POOLINGS:
            raise ValueError(
                f"pooling must be one of {self.POOLINGS}, got {pooling!r}."
            )
        self.hidden_dim = int(hidden_dim)
        self.summary_dim = int(summary_dim)
        self.pooling = pooling
        self.num_heads = int(num_heads)
        self.phi_1 = keras.layers.Dense(self.hidden_dim, activation="mish")
        self.phi_2 = keras.layers.Dense(self.hidden_dim)
        self.rho_1 = keras.layers.Dense(self.hidden_dim, activation="mish")
        self.rho_2 = keras.layers.Dense(self.summary_dim)
        if pooling in self._ATTENTION_POOLINGS:
            if self.hidden_dim % self.num_heads:
                raise ValueError("hidden_dim must be divisible by num_heads.")
            if pooling == "attention":
                self.query = keras.layers.Dense(self.hidden_dim)
            self.attention = keras.layers.MultiHeadAttention(
                num_heads=self.num_heads, key_dim=self.hidden_dim // self.num_heads
            )
            self.norm_1 = keras.layers.LayerNormalization()
            self.norm_2 = keras.layers.LayerNormalization()
            self.rff_1 = keras.layers.Dense(self.hidden_dim, activation="mish")
            self.rff_2 = keras.layers.Dense(self.hidden_dim)

    @property
    def output_dim(self) -> int:
        return self.summary_dim + 1

    @property
    def _pooled_dim(self) -> int:
        if self.pooling in ("mean_std", "mean_max"):
            return 2 * self.hidden_dim
        return self.hidden_dim

    def build(self, x_shape, h_shape):
        """Build from the shapes of the ``call`` arguments ``x`` and ``h``.

        The argument names match those of :meth:`call` (plus ``_shape``), as
        Keras requires for a multi-argument ``build``. The width of
        ``concat([x, h])`` and the attention query width are derived from them.
        """
        x_dim = int(x_shape[-1])
        query_dim = int(h_shape[-1])
        in_dim = x_dim + query_dim
        self.phi_1.build((None, in_dim))
        self.phi_2.build((None, self.hidden_dim))
        self.rho_1.build((None, self._pooled_dim))
        self.rho_2.build((None, self.hidden_dim))
        if self.pooling == "attention":
            self.query.build((None, None, query_dim))
        if self.pooling == "pma":
            self.seed = self.add_weight(
                shape=(self.hidden_dim,), initializer="glorot_uniform", name="seed"
            )
        if self.pooling in self._ATTENTION_POOLINGS:
            seq = (None, None, self.hidden_dim)
            self.attention.build(seq, seq, seq)
            self.norm_1.build(seq)
            self.norm_2.build(seq)
            self.rff_1.build(seq)
            self.rff_2.build(seq)
        self.ctx0 = self.add_weight(
            shape=(self.summary_dim,), initializer="zeros", name="ctx0"
        )
        super().build(x_shape)

    def phi_values(self, x, h):
        """Per-item set-element features ``phi(x_j, h_j)``."""
        return self.phi_2(self.phi_1(ops.concatenate([x, h], axis=-1)))

    def from_prefix_sum(self, prefix_sum, count):
        """Summary from a prefix sum of ``phi`` values and the prefix length.

        ``pooling="mean"`` only.

        Parameters
        ----------
        prefix_sum : Tensor
            ``(..., hidden_dim)`` sum of ``phi`` over the preceding items.
        count : Tensor
            ``(..., 1)`` number of preceding items (float).
        """
        mean = prefix_sum / ops.maximum(count, 1.0)
        return self._finish(mean, count)

    def _finish(self, pooled, count):
        ctx = self.rho_2(self.rho_1(pooled))
        ctx = ops.where(count < 0.5, self.ctx0, ctx)
        return ops.concatenate([ctx, ops.log(count + 1.0)], axis=-1)

    def _pool(self, values, mask, h_query):
        """Pool ``values`` ``(N, J, H)`` per query row under ``mask``.

        ``mask`` is ``(N, Q, J)`` (true = earlier draw) and ``h_query`` the
        ``(N, Q, D)`` embeddings of the items being drawn. Returns
        ``(N, Q, pooled_dim)``. Rows with an empty mask are finite but
        meaningless; ``_finish`` replaces them with ``ctx_0``.
        """
        if self.pooling in self._ATTENTION_POOLINGS:
            if self.pooling == "attention":
                q = self.query(h_query)
            else:  # one learned seed, the same query for every position
                q = ops.zeros_like(h_query[..., :1]) + self.seed
            att = self.attention(q, values, values, attention_mask=mask)
            hid = self.norm_1(q + att)
            return self.norm_2(hid + self.rff_2(self.rff_1(hid)))
        m = ops.expand_dims(ops.cast(mask, values.dtype), -1)  # (N, Q, J, 1)
        v = ops.expand_dims(values, 1)  # (N, 1, J, H)
        n_prev = ops.sum(m, axis=2)  # (N, Q, 1)
        mean = ops.sum(m * v, axis=2) / ops.maximum(n_prev, 1.0)
        if self.pooling == "mean":
            return mean
        if self.pooling == "mean_std":
            # centered second moment rather than E[p^2] - E[p]^2, which
            # cancels catastrophically when the mean is large
            centered = v - ops.expand_dims(mean, 2)
            var = ops.sum(m * ops.square(centered), axis=2) / ops.maximum(n_prev, 1.0)
            return ops.concatenate([mean, ops.sqrt(var + self._STD_EPS)], axis=-1)
        # mean_max: masked entries cannot win; empty rows give 0, not -inf
        mx = ops.max(ops.where(m > 0.5, v, ops.full_like(v, -1e9)), axis=2)
        mx = ops.where(n_prev > 0.5, mx, ops.zeros_like(mx))
        return ops.concatenate([mean, mx], axis=-1)

    def from_prefix(self, prefix, h_i, i: int):
        """Summary for one position from the stacked earlier ``phi`` values.

        Used at sampling time for every pooling.

        Parameters
        ----------
        prefix : Tensor or None
            ``(N, i, hidden_dim)`` ``phi`` values of the earlier draws, or
            ``None`` at position 0.
        h_i : Tensor
            ``(N, embed_dim)`` embedding of the item being drawn.
        i : int
            Position (number of earlier draws).
        """
        n = ops.shape(h_i)[0]
        count = ops.full((n, 1), float(i), dtype=h_i.dtype)
        if prefix is None:
            pooled = ops.zeros((n, self._pooled_dim), dtype=h_i.dtype)
        else:
            mask = ops.ones((n, 1, i), dtype="bool")
            pooled = self._pool(prefix, mask, ops.expand_dims(h_i, 1))[:, 0]
        return self._finish(pooled, count)

    def call(self, x, h):
        """Teacher-forced summaries for all positions.

        Parameters
        ----------
        x : Tensor
            ``(B, I, x_dim)`` true item parameters (network space).
        h : Tensor
            ``(B, I, embed_dim)`` item embeddings.

        Returns
        -------
        Tensor
            ``(B, I, summary_dim + 1)``; entry ``i`` depends only on
            ``x[:, :i]`` and ``h[:, :i]`` (for ``"attention"`` also on
            ``h[:, i]``, the query).
        """
        p = self.phi_values(x, h)
        # position index 0, 1, 2, ... as a float column, shape (B, I, 1)
        count = ops.cumsum(ops.ones_like(p[..., :1]), axis=1) - 1.0
        if self.pooling == "mean":
            csum = ops.cumsum(p, axis=1)
            prefix = ops.concatenate([ops.zeros_like(p[:, :1]), csum[:, :-1]], axis=1)
            return self.from_prefix_sum(prefix, count)
        n_items = ops.shape(p)[1]
        idx = ops.arange(n_items)
        causal = ops.expand_dims(idx, 0) < ops.expand_dims(idx, 1)  # [i, j]: j < i
        mask = ops.broadcast_to(
            ops.expand_dims(causal, 0), (ops.shape(p)[0], n_items, n_items)
        )
        return self._finish(self._pool(p, mask, h), count)

    def get_config(self):
        base = super().get_config()
        base.update(
            {
                "hidden_dim": self.hidden_dim,
                "summary_dim": self.summary_dim,
                "pooling": self.pooling,
                "num_heads": self.num_heads,
            }
        )
        return base
