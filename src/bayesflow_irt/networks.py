"""Custom summary networks for doubly-exchangeable (IRT) data.

Provides ``MatrixEquivariantIRTSummary``, which outputs per-item
embeddings ``(batch, num_items, embed_dim)`` and supports variable
person/item counts.
"""

from __future__ import annotations

import warnings
from importlib import import_module

import keras
import keras.ops as ops
from keras import layers

try:
    from bayesflow.networks.transformers.isab import (
        InducedSetAttentionBlock as InducedSetAttention,
    )
    from bayesflow.networks.transformers.pma import PoolingByMultiHeadAttention
    from bayesflow.networks.transformers.sab import SetAttentionBlock as SetAttention
except ModuleNotFoundError:
    _attention_pkg = "bayesflow.networks.summary.transformers.attention"
    _isab_module = import_module(f"{_attention_pkg}.induced_set_attention")
    _pma_module = import_module(f"{_attention_pkg}.pooling_by_multihead_attention")
    _sab_module = import_module(f"{_attention_pkg}.set_attention")
    InducedSetAttention = _isab_module.InducedSetAttention
    PoolingByMultiHeadAttention = _pma_module.PoolingByMultiHeadAttention
    SetAttention = _sab_module.SetAttention

# ---------------------------------------------------------------------------
# Shared helper: build a stack of SAB (or ISAB) layers
# ---------------------------------------------------------------------------


def _make_sab_stack(
    num_blocks: int,
    embed_dim: int,
    num_heads: int,
    mlp_depth: int,
    mlp_width: int,
    dropout: float,
    use_isab: bool = False,
    num_inducing_points: int = 32,
) -> list[keras.Layer]:
    """Return a list of SAB or ISAB layers."""
    block_cls = InducedSetAttention if use_isab else SetAttention
    stack = []
    for _ in range(num_blocks):
        kwargs = dict(
            embed_dim=embed_dim,
            num_heads=num_heads,
            mlp_depth=mlp_depth,
            mlp_width=mlp_width,
            dropout=dropout,
        )
        if use_isab:
            kwargs["num_inducing_points"] = num_inducing_points
        stack.append(block_cls(**kwargs))
    return stack


def _make_outer_item_layers(
    *,
    input_embed_dim: int,
    embed_dim: int,
    num_outer_sabs: int,
    outer_embed_dim: int,
    num_outer_heads: int,
    outer_mlp_depth: int,
    outer_mlp_width: int,
    dropout: float,
    use_outer_isab: bool,
    num_outer_inducing_points: int,
    layer_prefix: str = "",
) -> tuple[list[keras.Layer], keras.Layer | None, keras.Layer]:
    """Create shared outer-item encoder layers."""
    outer_sabs = _make_sab_stack(
        num_blocks=num_outer_sabs,
        embed_dim=outer_embed_dim,
        num_heads=num_outer_heads,
        mlp_depth=outer_mlp_depth,
        mlp_width=outer_mlp_width,
        dropout=dropout,
        use_isab=use_outer_isab,
        num_inducing_points=num_outer_inducing_points,
    )
    bridge_name = f"{layer_prefix}bridge" if layer_prefix else "bridge"
    output_name = f"{layer_prefix}output_proj" if layer_prefix else "output_proj"
    bridge = (
        layers.Dense(outer_embed_dim, name=bridge_name)
        if input_embed_dim != outer_embed_dim
        else None
    )
    output_proj = layers.Dense(embed_dim, name=output_name)
    return outer_sabs, bridge, output_proj


def _build_outer_item_layers(
    *,
    outer_sabs: list[keras.Layer],
    bridge: keras.Layer | None,
    output_proj: keras.Layer,
    num_items: int,
    input_embed_dim: int,
    outer_embed_dim: int,
    output_proj_input_dim: int | None = None,
) -> None:
    """Build shared outer-item encoder layers from static shape info.

    ``output_proj_input_dim`` overrides the input width ``output_proj`` is
    built with; it defaults to ``outer_embed_dim`` (the shared item state),
    but callers that insert extra layers between the outer SAB/ISAB stack and
    ``output_proj`` (e.g. parallel projection heads) pass the wider
    post-insertion width instead.
    """
    if bridge is not None:
        bridge.build((None, num_items, input_embed_dim))

    dummy_outer = ops.zeros((2, num_items, outer_embed_dim))
    for sab in outer_sabs:
        if not sab.built:
            if isinstance(sab, InducedSetAttention):
                sab(dummy_outer, training=False)
            else:
                sab.build((None, num_items, outer_embed_dim))

    if output_proj_input_dim is None:
        output_proj_input_dim = outer_embed_dim
    output_proj.build((None, num_items, output_proj_input_dim))


def _apply_outer_bridge_and_sabs(
    x,
    *,
    outer_sabs: list[keras.Layer],
    bridge: keras.Layer | None,
    training: bool,
):
    """Apply the shared outer-item bridge + SAB/ISAB stack (no output projection)."""
    if bridge is not None:
        x = bridge(x)
    for sab in outer_sabs:
        x = sab(x, training=training)
    return x


def _apply_outer_item_layers(
    x,
    *,
    outer_sabs: list[keras.Layer],
    bridge: keras.Layer | None,
    output_proj: keras.Layer,
    training: bool,
):
    """Apply shared outer-item bridge + SAB/ISAB stack + output projection."""
    x = _apply_outer_bridge_and_sabs(
        x, outer_sabs=outer_sabs, bridge=bridge, training=training
    )
    return output_proj(x)


#: Supported ``_HartfordEMNBlock.norm_placement`` values. See that class's
#: docstring for what each ordering computes. ``"pre_activation"`` is the
#: default; the others are experimental alternatives.
_NORM_PLACEMENTS = frozenset(
    {
        "pre_activation",
        "post_activation",
        "none",
        "residual_post_norm",
        "residual_pre_norm",
        "residual_no_norm",
        "per_branch",
    }
)

#: Placements whose residual connection requires ``input_dim == hidden_dim``.
_RESIDUAL_NORM_PLACEMENTS = frozenset(
    {"residual_post_norm", "residual_pre_norm", "residual_no_norm"}
)

#: Placements that own no ``LayerNormalization`` at all.
_NORM_FREE_PLACEMENTS = frozenset({"none", "residual_no_norm"})

#: Supported ``_HartfordEMNBlock.dropout_kind`` values. ``"element"`` is the
#: shipped default and is what both reference implementations actually run;
#: ``"channel"`` is what Hartford et al. (2018) §3.2 prescribe in prose; and
#: ``"row_col"`` is the scheme the reference code implements under the name
#: ``matrix_dropout``. See that class's docstring for the full discrepancy.
_DROPOUT_KINDS = frozenset({"element", "channel", "row_col"})


class _HartfordEMNBlock(keras.Layer):
    """Single Hartford exchangeable-matrix-network (EMN) block.

    Implements the permutation-equivariant linear basis from Hartford et al.
    (2018, ICML), Theorem 2.1 / Eq. (4), extended to multi-channel inputs.

    **Mathematical correspondence (Hartford et al., 2018)**

    For input tensor ``X`` with shape ``(B, P, I, K)`` — batch, persons
    (rows), items (columns), input channels — and output channel index ``o``,
    the Hartford multi-channel formula (Eq. 4) is:

    .. math::

        Y_{n,m}^{\\langle o \\rangle} = \\sigma\\!\\Bigl(
            \\sum_{k=1}^{K}\\bigl[
              w_1^{(k,o)} X_{n,m}^{(k)}
            + \\tfrac{w_2^{(k,o)}}{P} \\sum_{n'} X_{n',m}^{(k)}
            + \\tfrac{w_3^{(k,o)}}{I} \\sum_{m'} X_{n,m'}^{(k)}
            + \\tfrac{w_4^{(k,o)}}{PI} \\sum_{n',m'} X_{n',m'}^{(k)}
            \\bigr] + w_5^{\\langle o \\rangle}
        \\Bigr)

    The four pooling statistics and their layer names are:

    =========  =========================  ================  ===========
    Paper term  Pooling                   Layer attribute   Axis reduced
    =========  =========================  ================  ===========
    ``w₁``     self (identity)            ``self_proj``     —
    ``w₂``     column mean (per item)     ``col_mean_proj`` axis 1 (P)
    ``w₃``     row mean (per person)      ``row_mean_proj`` axis 2 (I)
    ``w₄``     global mean               ``global_proj``   axes 1 & 2
    ``w₅``     bias (per output channel)  ``bias``          —
    =========  =========================  ================  ===========

    .. note::
        **Naming convention:** ``row_mean_proj`` is applied to the mean
        *of each row* (averaging across items / axis 2), giving a per-person
        statistic.  ``col_mean_proj`` is applied to the mean *of each column*
        (averaging across persons / axis 1), giving a per-item statistic.
        In Hartford's notation with persons as rows (N) and items as columns
        (M), ``col_mean_proj`` corresponds to paper parameter ``w₂`` and
        ``row_mean_proj`` to ``w₃``.

    **Deviation from the paper**

    Hartford et al. apply the nonlinearity :math:`\\sigma` directly to the
    linear combination above.  This block instead uses
    ``linear combination → LayerNorm → activation``, i.e. the nonlinearity
    acts on the normalised pre-activations.  LayerNorm operates on the channel
    axis (axis -1) and is element-wise across the spatial axes (P, I), so it
    **preserves permutation equivariance**.  The deviation is a standard
    stabilisation technique for deep architectures and does not affect the
    theoretical PE guarantee.

    A further, weaker deviation concerns Theorem 2.1's hypothesis: it is
    stated for a *strictly monotonic* nonlinearity, while the default
    activation here is ``mish``, which dips below zero near ``x ≈ -1.19`` and
    so is not monotonic. This does not cost the equivariance guarantee. Strict
    monotonicity is needed for the theorem's *necessity* direction (PE implies
    this weight tying) and for Definition 2.1's full form (the layer must not
    be equivariant with respect to *any other* permutation of ``X``'s
    elements). The *sufficiency* direction actually relied on here — tied
    weights plus an element-wise nonlinearity implies permutation
    equivariance — holds for any element-wise activation. What weakens is the
    guarantee that the layer has no additional, unintended equivariances, not
    the equivariance the summary network depends on.

    A second deviation is the dropout granularity. Hartford et al. §3.2
    prescribe *channel* dropout for exchangeable matrix layers (whole
    channels dropped, equivalent to SpatialDropout); this block defaults to
    element-wise dropout. See ``dropout_kind`` for the trade-off and for the
    opt-in ``"channel"`` setting.

    **Sparse adaptation (not implemented)**

    For partially observed matrices Hartford et al. (Eq. 5) replace the
    fixed denominators ``P``, ``I``, and ``P*I`` with the actual counts of
    observed entries in each column, row, and globally.  The current
    implementation always uses dense means.  A sparse variant would be
    relevant for future package extensions supporting partially observed IRT
    response matrices (e.g., missing-at-random response data or adaptive
    testing designs).

    Parameters
    ----------
    hidden_dim : int
        Number of output channels (width of the block).
    activation : str
        Keras activation name applied after LayerNorm.
    dropout : float
        Dropout rate applied after the activation.  ``0.0`` disables dropout.
    dropout_kind : str
        **Experimental.** Granularity of the dropout mask.

        - ``"element"`` (default): an independent Bernoulli mask per
          ``(person, item, channel)`` entry — plain ``keras.layers.Dropout``.
          This is the shipped behavior.
        - ``"channel"``: whole channels are dropped, i.e. the mask is
          constant across the person and item axes
          (``noise_shape=(None, 1, 1, None)``). This is what Hartford et al.
          (2018) §3.2 prescribe for exchangeable matrix layers, on the
          grounds that "each channel in an exchangeable matrix layer is
          analogous to a single unit in a standard feed-forward network";
          they call it equivalent to *SpatialDropout* (Tompson et al., 2015,
          Sec. 3.2), which performs "only ``nfeats`` dropout trials and
          extend[s] the dropout value across the entire feature map" — the
          semantics ``noise_shape=(None, 1, 1, None)`` implements here.
        - ``"row_col"``: whole persons and whole items are dropped, in two
          independent draws, each tied across the channel axis
          (``noise_shape=(None, None, 1, 1)`` then ``(None, 1, None, 1)``).
          This is ``matrix_dropout`` in the reference implementation; in IRT
          terms it removes whole respondents and whole items from a training
          step.

        **The paper's prose and the paper's own code disagree, and this option
        set spans the disagreement.** Hartford et al. (2018) cite two official
        implementations (footnote 2). Neither implements the §3.2 prose:

        - TF ``channel_dropout_sparse`` reshapes to ``[-1, units]`` and calls
          ``tf.layers.dropout`` with no ``noise_shape`` — element-wise despite
          the name — and it is the variant actually enabled in the
          self-supervised ML-100K script.
        - TF ``matrix_dropout`` uses ``noise_shape=[N,1,1]`` then ``[1,M,1]``,
          i.e. rows and columns, not channels (that would be ``[1,1,K]``).
        - The PyTorch repo's training script uses plain ``nn.Dropout(p=0.5)``.

        So ``"element"`` is what the reference code runs, ``"channel"`` is what
        the paper text prescribes, and ``"row_col"`` is the third scheme the
        reference code actually provides. Which is best *here* is an empirical
        question, not a fidelity question — hence the option.

        Two reasons the distinction matters here. First, Tompson et al.'s own
        argument transfers directly: they found standard dropout ineffective
        for a fully convolutional network because strongly correlated
        activations feed *shared* weights, so a dropped position's zero
        gradient is swamped by its surviving correlated neighbours — "the
        effective learning rate is scaled by the dropout probability p, but
        independence is not enhanced". Every projection in this block is
        likewise shared across all ``(person, item)`` cells, so dropping
        individual cells rescales gradients rather than decorrelating them.
        Second, three of the four Hartford terms are means over the person
        and/or item axes, so an element mask is additionally averaged away in
        the *forward* pass at a rate of roughly ``O(dropout / sqrt(P))``, its
        effective strength decaying as the matrix grows. A channel mask has
        neither problem: it removes the channel from all four terms
        identically at any size. A channel mask is also constant
        across ``(P, I)`` and therefore commutes with row/column permutations
        *per draw*, making training-time permutation equivariance exact
        rather than merely distributional.

        All three kinds zero the same expected fraction of the tensor at a
        given rate (``row_col`` to first order in the rate); they differ in the
        variance and correlation of what is removed. Neither ``"channel"`` nor
        ``"row_col"`` is supported together with person chunking, because their
        masks are shared across the person and/or item axes and so must be
        drawn once for the whole matrix, while each chunk's dropout call draws
        its own.

        Experimental.
    norm_placement : str
        **Experimental.** Where the LayerNorm sits relative to the Hartford
        linear combination and the activation. The default reproduces the
        standard behavior exactly.

        - ``"pre_activation"`` (default): ``sum -> LN -> act``. The shipped
          ordering; see "Deviation from the paper" above.
        - ``"post_activation"``: ``sum -> act -> LN``. Normalizes the block's
          *output* (what the next block's projections see) rather than the
          activation's input.
        - ``"none"``: ``sum -> act``. Paper-faithful Hartford Eq. (4), and
          also faithful to both reference implementations, neither of which
          contains any normalization layer at all.
        - ``"residual_no_norm"``: ``x + act(sum)``. **The reference
          architecture.** The TF implementation applies the skip connection
          after the activation whenever the channel counts match
          (``if skip_connections and K == units: output = output + input``),
          and enables it in the factorized-autoencoder experiments. No
          normalization anywhere. This is the configuration the paper's
          reported results came from.
        - ``"residual_post_norm"``: ``LN(x + act(sum))``. The original
          Transformer post-norm residual ordering; matches the
          ``Add + LayerNorm`` structure of the outer SAB/ISAB stack.
        - ``"residual_pre_norm"``: ``x + act(sum(LN(x)))``. Pre-norm residual;
          the norm is applied to the block *input*, so it also changes the
          inputs to the row/column/global means.
        - ``"per_branch"``: each of the four projected Hartford terms is
          normalized separately before summation, then ``act``. Targets the
          size-dependent *relative* scale of the four terms (the pooled
          branches' variance shrinks with ``P``/``I``), which a single
          post-sum LN cannot correct.

        All placements act per ``(person, item)`` cell on the channel axis
        and therefore preserve permutation equivariance. The three
        ``residual_*`` placements require ``input_dim == hidden_dim``
        (validated in ``build``) and apply dropout before the add, so the
        identity path is never dropped and the three differ only in
        normalization. ``"residual_pre_norm"`` is not supported together with
        person chunking, because it changes the tensor the chunked
        ``col_mean``/``global_mean`` reductions must accumulate over;
        ``"residual_no_norm"`` has no such constraint.
    person_chunk_size : int or None
        If set and smaller than the input's person-axis length, compute the
        block via a two-pass person-chunked forward instead of materializing
        the full ``(B, P, I, H)`` tensor at once: a stats pass accumulates
        ``col_mean``/``global_mean`` (associative reductions over persons)
        via chunked sums, then an apply pass recombines each chunk's own
        ``row_mean`` with the finalized stats. This is numerically exact
        (floating-point tolerance only) and reduces to the dense path when
        ``None`` (default) or when the chunk size is not smaller than the
        actual number of persons. Forward-only: this path does not bound
        backward memory.

    References
    ----------
    .. [1] Hartford, J., Graham, D. R., Leyton-Brown, K., & Ravanbakhsh, S.
       (2018). Deep models of interactions across sets. *ICML 2018*, PMLR 80.
       Theorem 2.1 / Equations (3)–(5).
    """

    def __init__(
        self,
        hidden_dim: int,
        *,
        activation: str = "mish",
        dropout: float = 0.0,
        dropout_kind: str = "element",
        norm_placement: str = "pre_activation",
        person_chunk_size: int | None = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        if hidden_dim <= 0:
            raise ValueError("hidden_dim must be a positive integer.")
        if dropout < 0.0:
            raise ValueError("dropout must be non-negative.")
        if dropout_kind not in _DROPOUT_KINDS:
            raise ValueError(
                f"dropout_kind must be one of {sorted(_DROPOUT_KINDS)}, "
                f"got {dropout_kind!r}."
            )
        if dropout_kind in ("channel", "row_col") and person_chunk_size is not None:
            raise ValueError(
                f"dropout_kind={dropout_kind!r} is incompatible with person "
                "chunking: its mask is shared across the person and/or item "
                "axes, so it must be drawn once for the whole matrix, but each "
                "chunk's dropout call draws its own."
            )
        if norm_placement not in _NORM_PLACEMENTS:
            raise ValueError(
                f"norm_placement must be one of {sorted(_NORM_PLACEMENTS)}, "
                f"got {norm_placement!r}."
            )
        if person_chunk_size is not None and person_chunk_size <= 0:
            raise ValueError("person_chunk_size must be a positive integer or None.")
        if norm_placement == "residual_pre_norm" and person_chunk_size is not None:
            raise ValueError(
                "norm_placement='residual_pre_norm' is incompatible with person "
                "chunking: the norm changes the tensor the chunked col_mean/"
                "global_mean reductions accumulate over."
            )

        self.hidden_dim = hidden_dim
        self.activation = activation
        self.dropout = float(dropout)
        self.dropout_kind = dropout_kind
        self.norm_placement = norm_placement
        self.person_chunk_size = person_chunk_size

        self.self_proj = layers.Dense(hidden_dim, use_bias=False, name="self_proj")
        # col_mean_proj: projects the per-item mean (mean over persons, axis 1)
        # → Hartford w₂ term: (1/P) Σ_{n'} X_{n',m}^{(k)}
        self.col_mean_proj = layers.Dense(
            hidden_dim, use_bias=False, name="col_mean_proj"
        )
        # row_mean_proj: projects the per-person mean (mean over items, axis 2)
        # → Hartford w₃ term: (1/I) Σ_{m'} X_{n,m'}^{(k)}
        self.row_mean_proj = layers.Dense(
            hidden_dim, use_bias=False, name="row_mean_proj"
        )
        self.global_proj = layers.Dense(hidden_dim, use_bias=False, name="global_proj")
        # `norm` stays the single post-sum LayerNorm for every placement that
        # has one; `per_branch` instead owns four independent norms (one per
        # Hartford term) and leaves `norm` unused/None.
        if norm_placement == "per_branch":
            self.norm = None
            self.branch_norms = [
                layers.LayerNormalization(axis=-1, name=f"branch_norm_{term}")
                for term in ("self", "row_mean", "col_mean", "global")
            ]
        else:
            self.norm = (
                None
                if norm_placement in _NORM_FREE_PLACEMENTS
                else layers.LayerNormalization(axis=-1, name="norm")
            )
            self.branch_norms = None
        self.activation_layer = layers.Activation(activation, name="activation")
        # Inputs are (batch, P, I, H); `None` entries in noise_shape are
        # resolved from the input shape, so a literal 1 is what ties the mask
        # across an axis.
        #   element  -> mask varies everywhere (plain dropout).
        #   channel  -> (None, 1, 1, None): tied across persons and items, so a
        #               whole channel drops at once (Hartford §3.2 prose /
        #               SpatialDropout).
        #   row_col  -> two draws: (None, None, 1, 1) drops whole persons and
        #               (None, 1, None, 1) drops whole items, each tied across
        #               channels. This is `matrix_dropout` in the reference TF
        #               implementation. Two independent Keras Dropout layers
        #               each rescale by 1/(1-rate), giving 1/(1-rate)^2 overall,
        #               which is the correct inverted-dropout factor for a keep
        #               probability of (1-rate)^2. (The reference's sparse
        #               variant instead divides by 1-rate^2, which does not
        #               match its own keep probability.)
        if dropout <= 0.0:
            self.dropout_layer = None
            self.item_dropout_layer = None
        elif dropout_kind == "row_col":
            self.dropout_layer = layers.Dropout(
                dropout, noise_shape=(None, None, 1, 1), name="person_dropout"
            )
            self.item_dropout_layer = layers.Dropout(
                dropout, noise_shape=(None, 1, None, 1), name="item_dropout"
            )
        else:
            self.dropout_layer = layers.Dropout(
                dropout,
                noise_shape=(None, 1, 1, None) if dropout_kind == "channel" else None,
                name="dropout",
            )
            self.item_dropout_layer = None
        self.bias = None

    def build(self, input_shape):
        if len(input_shape) != 4:
            raise ValueError(
                f"Expected 4D input (batch, P, I, H), got shape {input_shape}."
            )
        in_dim = input_shape[-1]
        if self.norm_placement in _RESIDUAL_NORM_PLACEMENTS and in_dim is not None:
            if in_dim != self.hidden_dim:
                raise ValueError(
                    f"norm_placement={self.norm_placement!r} adds a residual "
                    f"connection and therefore requires input_dim == hidden_dim, "
                    f"got input_dim={in_dim} and hidden_dim={self.hidden_dim}."
                )
        self.self_proj.build(input_shape)
        self.row_mean_proj.build((None, None, 1, in_dim))
        self.col_mean_proj.build((None, 1, None, in_dim))
        self.global_proj.build((None, 1, 1, in_dim))
        self.bias = self.add_weight(
            name="bias",
            shape=(self.hidden_dim,),
            initializer="zeros",
            trainable=True,
        )
        if self.norm is not None:
            self.norm.build((None, None, None, self.hidden_dim))
        if self.branch_norms is not None:
            # Each branch norm sees the *projected* term, so all four are
            # hidden_dim-wide; their spatial axes differ (broadcast shapes)
            # but LayerNormalization(axis=-1) only sizes weights from the
            # channel axis.
            for branch_norm in self.branch_norms:
                branch_norm.build((None, None, None, self.hidden_dim))
        super().build(input_shape)

    def call(self, x, training=False, **kwargs):
        n_persons = x.shape[1]
        if (
            self.person_chunk_size is not None
            and n_persons is not None
            and self.person_chunk_size < n_persons
        ):
            return self._chunked_call(x, training=training)

        # For the pre-norm residual placement the norm applies to the block
        # *input*, so the pooled means are taken over the normalized tensor;
        # the residual branch still carries the un-normalized input.
        x_in = self.norm(x) if self.norm_placement == "residual_pre_norm" else x

        # row_mean: mean across items (axis 2) → per-person statistic (Hartford w₃)
        row_mean = ops.mean(x_in, axis=2, keepdims=True)
        # col_mean: mean across persons (axis 1) → per-item statistic (Hartford w₂)
        col_mean = ops.mean(x_in, axis=1, keepdims=True)
        global_mean = ops.mean(x_in, axis=(1, 2), keepdims=True)

        # Hartford Eq. (4) linear combination, then the placement-dependent
        # norm/activation epilogue (paper applies σ directly to the sum; the
        # default adds LN before it — see class docstring).
        y = self._combine_terms(
            self.self_proj(x_in),
            self.row_mean_proj(row_mean),
            self.col_mean_proj(col_mean),
            self.global_proj(global_mean),
        )
        return self._epilogue(y, x, training=training)

    def _combine_terms(self, self_term, row_term, col_term, global_term):
        """Sum the four projected Hartford terms and add the bias.

        Under ``norm_placement="per_branch"`` each term is normalized on its
        own first, so the four branches enter the sum at comparable scale
        regardless of ``P``/``I``; every other placement sums them directly
        as in Hartford Eq. (4).
        """
        if self.branch_norms is not None:
            self_norm, row_norm, col_norm, global_norm = self.branch_norms
            self_term = self_norm(self_term)
            row_term = row_norm(row_term)
            col_term = col_norm(col_term)
            global_term = global_norm(global_term)
        return self_term + row_term + col_term + global_term + self.bias

    def _epilogue(self, y, residual_input, training=False):
        """Apply the placement-dependent norm/activation/dropout tail.

        ``y`` is the summed linear combination and ``residual_input`` the
        block input the ``residual_*`` placements add back. Shared by the
        dense ``call`` and the chunked ``_apply_chunk`` so both paths get
        identical semantics for every placement.
        """
        placement = self.norm_placement
        if placement == "pre_activation":
            y = self.activation_layer(self.norm(y))
        elif placement == "post_activation":
            y = self.norm(self.activation_layer(y))
        else:
            y = self.activation_layer(y)

        if placement in _RESIDUAL_NORM_PLACEMENTS:
            # Transformer convention: dropout applies to the sublayer output
            # *before* the residual add, so the identity path is never
            # dropped. Held fixed across all three residual placements so the
            # options differ in normalization, not dropout position.
            y = self._apply_dropout(y, training=training)
            y = residual_input + y
            if placement == "residual_post_norm":
                y = self.norm(y)
            return y

        return self._apply_dropout(y, training=training)

    def _apply_dropout(self, y, training=False):
        """Apply the configured dropout draws, if any.

        ``row_col`` needs two draws (persons, then items); every other kind
        needs one. Keeping this in one place means the dense and chunked
        paths cannot disagree about how many masks were drawn.
        """
        if self.dropout_layer is not None:
            y = self.dropout_layer(y, training=training)
        if self.item_dropout_layer is not None:
            y = self.item_dropout_layer(y, training=training)
        return y

    def _chunked_call(self, x, training=False):
        """Two-pass person-chunked forward; exact match to the dense path.

        ``col_mean``/``global_mean`` are associative reductions over persons
        (chunked-summable), while
        ``row_mean`` has no cross-person dependency (computable per chunk).
        Forward-only: gradients flow through this path today the same as
        through the dense path (no checkpointing here yet), so it currently
        trades peak *activation* memory for the same total compute, not
        gradient memory; ``_chunked_matrix_and_pool_trainable`` bounds
        backward memory. ``x`` is already fully accelerator-resident here, so
        (unlike ``MatrixEquivariantIRTSummary``'s CPU-staged cross-block
        forward) the final ``concatenate`` still produces an
        ``O(n_persons)``-sized output — this method only bounds *transient*
        per-chunk activation memory, not the final output size.
        """
        n_persons = x.shape[1]
        n_items = x.shape[2]
        chunk_size = self.person_chunk_size
        starts = list(range(0, n_persons, chunk_size))
        chunks = [x[:, s : min(s + chunk_size, n_persons)] for s in starts]

        col_term, global_term = self._stats_terms(chunks, n_persons, n_items)
        outputs = [
            self._apply_chunk(c, col_term, global_term, training) for c in chunks
        ]
        return ops.concatenate(outputs, axis=1)

    def _stats_terms(self, chunks, n_persons, n_items, return_means=False):
        """Stats pass: accumulate ``col_term``/``global_term`` over chunks.

        ``chunks`` may be a list (chunk-view source already fully resident,
        as in ``_chunked_call``) or a generator (chunk source materialized
        lazily one at a time, as in
        ``MatrixEquivariantIRTSummary``'s CPU-staged forward) — this
        method only ever holds one chunk's running sum plus the current
        chunk, so both call sites get the same peak-memory shape regardless
        of which iterable they pass.

        Accumulates in at least float32: float16/bfloat16 chunk sums can
        overflow well before the (finite) mean is reached -- e.g. summing
        100,000 float16 ones already exceeds float16's ~65,504 max
        representable value, even though ``ops.mean`` on the same data (the
        dense path) is unaffected. Only the accumulation needs widening; the
        finalized means are cast back to the original dtype before the
        projections, matching what the dense path would produce.

        ``return_means=True`` additionally returns the pre-projection
        ``col_mean``/``global_mean`` values, needed by
        ``MatrixEquivariantIRTSummary``'s trainable chunked backward
        (the trainable chunked path): that mechanism re-derives
        ``col_term``/``global_term``
        from a detached leaf built from these raw means, rather than from
        this call's own (graph-attached) projections.
        """
        col_sum = None
        global_sum = None
        orig_dtype = None
        accum_dtype = None
        for chunk in chunks:
            if orig_dtype is None:
                orig_dtype = chunk.dtype
                accum_dtype = (
                    "float32"
                    if keras.backend.standardize_dtype(orig_dtype)
                    in ("float16", "bfloat16")
                    else orig_dtype
                )
            chunk = ops.cast(chunk, accum_dtype)
            chunk_col_sum = ops.sum(chunk, axis=1, keepdims=True)
            chunk_global_sum = ops.sum(chunk, axis=(1, 2), keepdims=True)
            col_sum = chunk_col_sum if col_sum is None else col_sum + chunk_col_sum
            global_sum = (
                chunk_global_sum
                if global_sum is None
                else global_sum + chunk_global_sum
            )
        col_mean = ops.cast(col_sum / n_persons, orig_dtype)
        global_mean = ops.cast(global_sum / (n_persons * n_items), orig_dtype)
        if return_means:
            return col_mean, global_mean
        return self.col_mean_proj(col_mean), self.global_proj(global_mean)

    def _apply_chunk(self, chunk, col_term, global_term, training):
        """Apply pass for one chunk, given the finalized stats-pass terms."""
        row_mean = ops.mean(chunk, axis=2, keepdims=True)
        y = self._combine_terms(
            self.self_proj(chunk),
            self.row_mean_proj(row_mean),
            col_term,
            global_term,
        )
        return self._epilogue(y, chunk, training=training)

    def get_config(self):
        base = super().get_config()
        return base | {
            "hidden_dim": self.hidden_dim,
            "activation": self.activation,
            "dropout": self.dropout,
            "dropout_kind": self.dropout_kind,
            "norm_placement": self.norm_placement,
            "person_chunk_size": self.person_chunk_size,
        }


# ---------------------------------------------------------------------------
# Shared helper: gradient-checkpointed forward wrapper
# ---------------------------------------------------------------------------


def _checkpoint_if_enabled(
    fn,
    x,
    training: bool,
    *,
    gradient_checkpointing: bool,
    backend_is_torch: bool,
    extra_guard: bool = True,
):
    """Run *fn* under gradient checkpointing when enabled and supported.

    Checkpointing is applied only when *all* of the following hold:

    1. ``gradient_checkpointing`` is ``True``
    2. ``training`` is ``True``
    3. the active Keras backend is PyTorch (``backend_is_torch``)
    4. *extra_guard* (a caller-supplied condition, e.g. "there is at least
       one block to checkpoint")

    Otherwise *fn* is called directly on ``(x, training)``.
    """
    use_cp = gradient_checkpointing and training and backend_is_torch and extra_guard
    if not use_cp:
        return fn(x, training)

    import torch.utils.checkpoint as cp

    # use_reentrant=True: avoids CheckpointError from BayesFlow ISAB layers
    # saving a non-deterministic number of tensors with the stricter
    # use_reentrant=False API. Deprecated but functional.
    #
    # Reentrant checkpointing only sets up backward hooks when at least one
    # input has requires_grad=True. The data tensor x never requires grad,
    # so without this the entire wrapped forward silently receives zero
    # gradients. We enable grad on a view so that checkpoint wires up the
    # recomputation graph; the spurious data gradient is never read.
    if not x.requires_grad:
        x = x.detach().requires_grad_(True)
    return cp.checkpoint(fn, x, training, use_reentrant=True)


def _widened_person_mean(chunks, n_persons, to_device=None):
    """Person-axis mean over already person-summed-per-chunk chunks,
    accumulating in at least float32 to avoid float16/bfloat16 overflow.

    Shared by ``MatrixEquivariantIRTSummary._chunked_matrix_and_pool``
    and ``_ChunkedTrainableMatrixPool.forward``'s final
    person-pooling reduction. Both stage the last matrix block's per-chunk
    output (to host memory or CPU) and need to average it over the full
    person axis without ever materializing the whole ``(B, n_persons, I,
    H)`` tensor at once — same shape of problem as
    ``_HartfordEMNBlock._stats_terms``'s ``col_mean``/``global_mean``
    accumulation, and the same fix: summing many same-signed, non-tiny
    chunks in float16 can overflow float16's ~65,504 max well before the
    (finite) mean is reached, even though the dense ``ops.mean`` on the same
    data is unaffected (10 float16 chunks totaling 70,000 ones sum to
    ``inf`` here, while dense ``ops.mean`` returns ``1.0``).

    Parameters
    ----------
    chunks
        Iterable of per-chunk arrays/tensors along the person axis (axis 1).
        May be NumPy arrays (inference path) or CPU-resident torch tensors
        (trainable path); ``to_device``, if given, moves each chunk before summing.
    n_persons : int
        Total person count across all chunks (the divisor for the mean).
    to_device : callable or None
        Optional ``chunk -> chunk`` transform applied before converting to a
        tensor (e.g. moving a CPU-staged torch tensor to the accelerator).
    """
    pooled_sum = None
    orig_dtype = None
    accum_dtype = None
    for chunk in chunks:
        chunk = ops.convert_to_tensor(chunk)
        if to_device is not None:
            chunk = to_device(chunk)
        if orig_dtype is None:
            orig_dtype = chunk.dtype
            accum_dtype = (
                "float32"
                if keras.backend.standardize_dtype(orig_dtype)
                in ("float16", "bfloat16")
                else orig_dtype
            )
        chunk = ops.cast(chunk, accum_dtype)
        chunk_sum = ops.sum(chunk, axis=1)
        pooled_sum = chunk_sum if pooled_sum is None else pooled_sum + chunk_sum
    return ops.cast(pooled_sum / n_persons, orig_dtype)


_CHUNKED_TRAINABLE_MATRIX_POOL_FN = None


def _get_chunked_trainable_matrix_pool_fn():
    """Lazily build (and cache) the ``torch.autograd.Function`` for the
    gradient-correct, memory-bounded person-chunked matrix-block forward.

    A "checkpointed forward + per-chunk immediate backward" strategy
    only bounds backward memory when a block's input has no upstream
    trainable parameters feeding it (the isolated single-block case). The
    real network chains a trainable ``cell_proj`` into
    1+ trainable ``_HartfordEMNBlock``s, so every block's input *does* have
    upstream trainable parameters -- and per-chunk ``backward(retain_graph=
    True)`` in that case must keep the *entire* stats-pass graph (every
    chunk's contribution to ``col_sum``/``global_sum``) alive until the last
    chunk's backward call, since gradient must reach every summand. That is
    ``O(n_persons)`` memory, not ``O(chunk_size)``, even though the computed
    *values* are correct (accumulation via ``retain_graph`` does not double
    count).

    The fix: treat each block's ``col_mean``/``global_mean`` (and the final
    person-pooled output) as an explicit *detach point* -- a fresh
    ``requires_grad_()`` leaf built from the numeric value already computed
    during a ``torch.no_grad()`` forward-staging sweep (the same CPU-staging
    ``MatrixEquivariantIRTSummary._chunked_matrix_and_pool`` already uses,
    the inference path -- this reuses those staged per-chunk values as the backward
    pass's recompute cache). Backward then proceeds in three passes per
    block, chunk-independent and ``O(chunk_size)`` peak memory each: (1) a
    cheap backward through whatever sits downstream of the pooled output,
    scattered uniformly into a per-chunk seed (pooling is a plain mean); (2)
    a checkpointed-equivalent per-chunk apply-pass backward (immediate,
    ``retain_graph`` only within *this block's own* small leaf-based
    ``col_term``/``global_term`` node -- not the whole stats-pass graph);
    (3) once every chunk has contributed to that leaf's accumulated
    gradient, a fully independent (no ``retain_graph`` needed) per-chunk
    stats-pass backward using that now-known gradient as a fixed seed. Doing
    (2)+(3) for each block from last to first, chaining the resulting
    per-chunk seed down to the previous block (or ``cell_proj``), propagates
    gradients the whole way down in ``O(2 x n_blocks x n_chunks)`` total
    compute (linear, not quadratic) with ``O(chunk_size)`` peak memory.

    Supports nonzero ``dropout`` on the matrix blocks. Dropout's mask is
    drawn once during the forward staging sweep (to produce the pooled
    output that actually feeds the loss); recomputing a chunk's apply pass
    during backward would otherwise draw a *different* mask, silently
    computing gradients for a different realized forward pass than the one
    that produced the loss. ``forward`` avoids this by snapshotting the CPU
    and (if applicable) CUDA RNG state immediately before each
    ``(block, chunk)``'s only stochastic op (``_apply_chunk``'s
    ``dropout_layer`` call); ``backward`` restores that exact snapshot
    before recomputing the same call, so the recomputed mask always matches
    the one used in the forward pass that produced the loss. This is the
    same RNG-preservation technique ``torch.utils.checkpoint``'s
    ``preserve_rng_state=True`` uses for the analogous problem (see
    ``torch.utils.checkpoint.get_device_states``/``set_device_states``);
    dropout here goes through Keras's ``layers.Dropout``, but on the torch
    backend with an unseeded ``SeedGenerator`` (the default -- no explicit
    ``seed=`` passed anywhere in this file), it resolves to
    ``torch.nn.functional.dropout``'s "fast path", which consumes torch's
    ordinary global RNG state rather than Keras's own counter-based
    ``SeedGenerator`` state -- verified against
    ``keras.src.backend.torch.random.dropout`` before relying on it, since
    the two mechanisms are not interchangeable and restoring the wrong one
    would silently reproduce a different mask.

    Two consequences of recomputing dropout in ``backward``, both handled
    the way ``torch.utils.checkpoint(preserve_rng_state=True)`` handles
    them:

    * ``backward`` also snapshots the caller's RNG state on entry and
      restores it before returning. Otherwise the per-chunk ``set_rng_state``
      rewinds would leave torch's global RNG at the last recompute's state
      rather than where the caller left it after the forward pass, letting a
      later training step reuse random numbers this backward already consumed.
    * The snapshot only covers CPU and CUDA. Other torch devices with their
      own RNG (notably Apple MPS) are *not* captured, so
      ``_can_use_trainable_chunking`` restricts this path to CPU/CUDA
      whenever any block has nonzero dropout; other devices fall back to the
      dense path. With zero dropout there is no stochastic op to reproduce,
      so any torch device is eligible.

    ``x_needs_grad``: ``x`` is always
    forced ``requires_grad_(True)`` by the caller so this Function's output
    correctly hooks into the outer autograd graph, but that does not mean
    ``x``'s own gradient is unneeded -- if ``x`` is itself produced by some
    upstream trainable layer (not the typical case, where ``x`` is raw
    response data, but not prevented by anything either), the dense path
    *would* propagate a gradient into that layer, and silently returning
    ``None`` here instead would leave it untrained without any error.
    ``x_needs_grad`` records whether ``x`` *genuinely* required grad before
    that forcing workaround; ``backward`` only pays for assembling and
    returning ``x``'s real (necessarily ``O(n_persons)``-sized) gradient
    when it is set.
    """
    global _CHUNKED_TRAINABLE_MATRIX_POOL_FN
    if _CHUNKED_TRAINABLE_MATRIX_POOL_FN is not None:
        return _CHUNKED_TRAINABLE_MATRIX_POOL_FN

    import torch

    def _last_chunk_true_width(starts, chunk_size, n_persons):
        # All chunks except possibly the last are exactly chunk_size wide
        # (starts steps by chunk_size); the last is n_persons - starts[-1],
        # in [1, chunk_size].
        return n_persons - starts[-1]

    def _padded_slice(src, start, chunk_size, n_persons):
        # Every chunk -- including the last -- is sliced to exactly
        # chunk_size elements, zero-padding past n_persons if needed. This
        # keeps every step's person-chunk tensor shape identical regardless
        # of the (randomly varying, per-batch) n_persons % chunk_size
        # remainder, so PyTorch's CUDA caching allocator sees a small,
        # repeating set of block sizes instead of a genuinely novel shape on
        # almost every step -- see ``utils._PeriodicCudaCacheRelease`` on
        # the allocator-reserved-memory ratchet this avoids. Padding rows
        # are person-count-agnostic zeros; every downstream sum/mean already
        # divides by the true n_persons (never a padded count), so they only
        # affect the result if left unmasked after passing through a
        # bias-carrying layer (cell_proj, _apply_chunk) -- see _mask_pad.
        end = start + chunk_size
        if end <= n_persons:
            return src[:, start:end]
        real = src[:, start:n_persons]
        pad_len = end - n_persons
        pad = real.new_zeros((real.shape[0], pad_len) + tuple(real.shape[2:]))
        return torch.cat([real, pad], dim=1)

    def _mask_pad(t, chunk_idx, n_chunks, true_width, chunk_size):
        # Zero out a chunk's padding rows (present only on the last chunk,
        # only when true_width < chunk_size). Necessary because cell_proj and
        # _apply_chunk carry bias terms, so a merely-zero-padded *input* row
        # does not generally produce a zero *output* row -- left unmasked,
        # padding rows would silently leak nonzero, person-count-invariant
        # bias values into col_mean/global_mean/the pooled output. Done via
        # an out-of-place multiply (not in-place slice assignment) so it
        # stays safe to call on tensors that participate in autograd.
        if chunk_idx != n_chunks - 1 or true_width == chunk_size:
            return t
        mask_shape = (1, chunk_size) + (1,) * (t.dim() - 2)
        mask = t.new_ones(mask_shape)
        mask[:, true_width:] = 0.0
        return t * mask

    class _ChunkedTrainableMatrixPool(torch.autograd.Function):
        @staticmethod
        def forward(ctx, x, summary_net, chunk_size, training, x_needs_grad):
            n_persons = x.shape[1]
            starts = list(range(0, n_persons, chunk_size))
            n_chunks = len(starts)
            last_true_width = _last_chunk_true_width(starts, chunk_size, n_persons)
            device = x.device

            # rng_states[block_idx][chunk_idx] = (cpu_state, cuda_state)
            rng_states = []
            with torch.no_grad():
                staged0 = [
                    _mask_pad(
                        summary_net.cell_proj(
                            _padded_slice(x, s, chunk_size, n_persons)
                        ),
                        i,
                        n_chunks,
                        last_true_width,
                        chunk_size,
                    ).cpu()
                    for i, s in enumerate(starts)
                ]
                staged = [staged0]
                block_stats = []
                n_items = staged0[0].shape[2]

                for block in summary_net.matrix_blocks:
                    prev = staged[-1]
                    col_mean, global_mean = block._stats_terms(
                        (c.to(device) for c in prev),
                        n_persons=n_persons,
                        n_items=n_items,
                        return_means=True,
                    )
                    block_stats.append((col_mean.cpu(), global_mean.cpu()))
                    col_term = block.col_mean_proj(col_mean)
                    global_term = block.global_proj(global_mean)

                    block_rng_states = []
                    out_chunks = []
                    for i, c in enumerate(prev):
                        # Snapshot the RNG state immediately before the only
                        # stochastic op _apply_chunk can run (dropout, when
                        # block.dropout > 0 -- see _apply_chunk). backward()
                        # restores this exact state before recomputing the
                        # same call, so the recompute reproduces the same
                        # dropout mask that produced pooled_value/the loss
                        # here, instead of drawing a fresh, different mask
                        # (see _get_chunked_trainable_matrix_pool_fn's
                        # docstring). Cheap regardless of block.dropout, so
                        # captured unconditionally rather than branching.
                        block_rng_states.append(
                            (
                                torch.get_rng_state(),
                                torch.cuda.get_rng_state(device)
                                if device.type == "cuda"
                                else None,
                            )
                        )
                        out_chunks.append(
                            _mask_pad(
                                block._apply_chunk(
                                    c.to(device), col_term, global_term, training
                                ),
                                i,
                                n_chunks,
                                last_true_width,
                                chunk_size,
                            ).cpu()
                        )
                    rng_states.append(block_rng_states)
                    staged.append(out_chunks)

                pooled_value = _widened_person_mean(
                    staged[-1], n_persons, to_device=lambda c: c.to(device)
                )

            ctx.save_for_backward(x)
            ctx.summary_net = summary_net
            ctx.chunk_size = chunk_size
            ctx.training = training
            # staged[-1] (the last matrix block's own apply-pass output) was
            # only needed to compute pooled_value just above; backward's
            # `for b in reversed(range(n_blocks))` loop only ever reads
            # staged[b] for b < n_blocks (cell_proj's output plus every
            # block's output *except* the last -- the last block's own
            # output is never re-read on the way back down). Keeping it in
            # ctx would hold an unused O(n_persons x n_items x H) CPU array
            # alive for the entire backward pass.
            ctx.staged = staged[:-1]
            ctx.block_stats = block_stats
            ctx.rng_states = rng_states
            ctx.n_persons = n_persons
            ctx.n_items = n_items
            ctx.device = device
            ctx.x_needs_grad = x_needs_grad
            return pooled_value.detach().clone()

        @staticmethod
        def backward(ctx, grad_pooled_output):
            (x,) = ctx.saved_tensors
            summary_net = ctx.summary_net
            chunk_size = ctx.chunk_size
            training = ctx.training
            staged = ctx.staged
            block_stats = ctx.block_stats
            rng_states = ctx.rng_states
            n_persons = ctx.n_persons
            n_items = ctx.n_items
            device = ctx.device
            x_needs_grad = ctx.x_needs_grad

            # Snapshot the caller's RNG state *before* any of the per-chunk
            # set_rng_state() rewinds below run. Restoring each saved forward
            # snapshot reproduces the right dropout mask for the recompute,
            # but it also leaves torch's global RNG stream sitting at the last
            # recompute's state when backward returns -- not where the caller
            # left it after the forward pass. Unlike the dense path (whose
            # backward consumes no RNG at all), that would let the *next*
            # training step draw dropout masks from random numbers this
            # backward already advanced past, silently correlating masks
            # across steps. torch.utils.checkpoint(preserve_rng_state=True)
            # avoids the identical problem by bracketing its recomputation
            # with a caller-state save/restore; this custom path needs the
            # same protection, so restore this snapshot before returning.
            caller_cpu_rng_state = torch.get_rng_state()
            caller_cuda_rng_state = (
                torch.cuda.get_rng_state(device) if device.type == "cuda" else None
            )

            starts = list(range(0, n_persons, chunk_size))
            n_chunks = len(starts)
            n_blocks = len(summary_net.matrix_blocks)
            last_true_width = _last_chunk_true_width(starts, chunk_size, n_persons)

            # pooled = (1/n_persons) * sum_p y_p => d(pooled)/d(y_p) =
            # 1/n_persons for every real person p in every chunk -- every
            # chunk gets the same per-item seed, broadcast over its person
            # axis, uniformly chunk_size wide to match forward's padded
            # chunk shapes (see _padded_slice). The last chunk's padding
            # rows get a zero seed, not the broadcast value: unlike a merely
            # zero-valued *activation*, a zero *seed* correctly propagates
            # zero gradient contribution regardless of what nonzero value
            # _apply_chunk's bias terms produce at those (fake, discarded)
            # rows during the recompute below -- see _mask_pad.
            seed_cpu = (grad_pooled_output / n_persons).detach().cpu()
            seed_per_chunk_cpu = [
                _mask_pad(
                    seed_cpu.unsqueeze(1).expand(-1, chunk_size, -1, -1).clone(),
                    i,
                    n_chunks,
                    last_true_width,
                    chunk_size,
                )
                for i in range(n_chunks)
            ]

            for b in reversed(range(n_blocks)):
                block = summary_net.matrix_blocks[b]
                col_mean_val, global_mean_val = block_stats[b]
                col_mean_leaf = (
                    col_mean_val.to(device).clone().detach().requires_grad_(True)
                )
                global_mean_leaf = (
                    global_mean_val.to(device).clone().detach().requires_grad_(True)
                )
                with torch.enable_grad():
                    col_term = block.col_mean_proj(col_mean_leaf)
                    global_term = block.global_proj(global_mean_leaf)

                direct_seed_cpu = []
                for i, start in enumerate(starts):
                    chunk_in = staged[b][i].to(device).detach().requires_grad_(True)
                    seed_gpu = seed_per_chunk_cpu[i].to(device)
                    # Restore the exact RNG state forward() captured before
                    # this (block, chunk)'s _apply_chunk call, so recomputing
                    # it here draws the identical dropout mask that produced
                    # pooled_value/the loss -- not a fresh, different one.
                    cpu_rng_state, cuda_rng_state = rng_states[b][i]
                    torch.set_rng_state(cpu_rng_state)
                    if cuda_rng_state is not None:
                        torch.cuda.set_rng_state(cuda_rng_state, device)
                    with torch.enable_grad():
                        y = block._apply_chunk(
                            chunk_in, col_term, global_term, training
                        )
                        is_last = i == n_chunks - 1
                        y.backward(gradient=seed_gpu, retain_graph=not is_last)
                    direct_seed_cpu.append(chunk_in.grad.cpu())
                    del chunk_in, y, seed_gpu

                # col_mean_leaf.grad/global_mean_leaf.grad are now fully
                # accumulated (every chunk's apply-pass backward touched the
                # shared col_term/global_term node). Propagate into the
                # stats pass, independently per chunk -- no retain_graph
                # needed, this seed is already a fixed, non-graph tensor.
                col_seed = col_mean_leaf.grad / n_persons
                global_seed = global_mean_leaf.grad / (n_persons * n_items)

                combined_seed_cpu = []
                for i, start in enumerate(starts):
                    chunk_in = staged[b][i].to(device).detach().requires_grad_(True)
                    with torch.enable_grad():
                        chunk_col_sum = chunk_in.sum(dim=1, keepdim=True)
                        chunk_global_sum = chunk_in.sum(dim=(1, 2), keepdim=True)
                        torch.autograd.backward(
                            [chunk_col_sum, chunk_global_sum],
                            grad_tensors=[
                                col_seed.expand_as(chunk_col_sum),
                                global_seed.expand_as(chunk_global_sum),
                            ],
                        )
                    # col_seed/global_seed are per-batch scalars broadcast
                    # uniformly over every row of chunk_in -- including the
                    # last chunk's padding rows, which are not real persons.
                    # Left unmasked, that broadcast leaks a nonzero seed into
                    # the padding-row positions of combined_seed_cpu, which
                    # then becomes the *next* (earlier) block's incoming
                    # seed: since staged[b-1][i]'s padding rows are also
                    # nonzero after _apply_chunk's bias (only their
                    # *forward* output was masked at block b, not b-1), a
                    # leaked nonzero seed there would inject a real,
                    # incorrect contribution into that earlier block's
                    # col_mean_leaf/global_mean_leaf gradient -- not just an
                    # eventually-discarded chunk_in.grad. Re-mask before
                    # propagating.
                    combined = _mask_pad(
                        direct_seed_cpu[i].to(device) + chunk_in.grad,
                        i,
                        n_chunks,
                        last_true_width,
                        chunk_size,
                    )
                    combined_seed_cpu.append(combined.cpu())
                    del chunk_in, combined

                seed_per_chunk_cpu = combined_seed_cpu

            # x_needs_grad reflects whether x *originally* required grad,
            # before _chunked_matrix_and_pool_trainable's workaround forced
            # requires_grad_(True) on it purely to get this Function's
            # output correctly hooked into the outer autograd graph (see
            # that method's docstring). If x is genuinely a leaf of some
            # upstream trainable computation (e.g. a learned preprocessing
            # layer feeding this summary network), returning None here would
            # silently drop gradient into that layer -- a correctness bug
            # the dense path does not have. Only pay for assembling the full x-shaped
            # gradient (itself O(n_persons)-sized, an inherent cost of
            # needing it at all) when it is actually going to be consumed.
            grad_x_chunks_cpu = [] if x_needs_grad else None
            for i, start in enumerate(starts):
                end = min(start + chunk_size, n_persons)
                raw_chunk = x[:, start:end].to(device).detach()
                if x_needs_grad:
                    raw_chunk.requires_grad_(True)
                # x is the true (unpadded) input, so raw_chunk is only
                # true-width wide for the last chunk -- truncate the
                # (uniformly chunk_size-wide, zero-padded-tail) seed to
                # match. Padding rows are already zero, so this discards
                # nothing but the padding itself.
                seed_gpu = seed_per_chunk_cpu[i][:, : end - start].to(device)
                with torch.enable_grad():
                    out = summary_net.cell_proj(raw_chunk)
                    out.backward(gradient=seed_gpu)
                if x_needs_grad:
                    grad_x_chunks_cpu.append(raw_chunk.grad.cpu())
                    del raw_chunk

            grad_x = None
            if x_needs_grad:
                grad_x = torch.cat(grad_x_chunks_cpu, dim=1).to(x.device)

            # Restore the caller's pre-backward RNG state (see the snapshot at
            # the top of backward): recomputation is an implementation detail
            # of this Function's gradient and must not perturb the global RNG
            # stream the surrounding training loop observes.
            torch.set_rng_state(caller_cpu_rng_state)
            if caller_cuda_rng_state is not None:
                torch.cuda.set_rng_state(caller_cuda_rng_state, device)

            # Gradients for (x, summary_net, chunk_size, training,
            # x_needs_grad): every real trainable parameter's gradient was
            # already accumulated above as an ordinary side effect of the
            # nested backward() calls (they reference the real
            # cell_proj/block parameter objects directly, not a copy routed
            # through this Function's input/output contract).
            return grad_x, None, None, None, None

    _CHUNKED_TRAINABLE_MATRIX_POOL_FN = _ChunkedTrainableMatrixPool
    return _CHUNKED_TRAINABLE_MATRIX_POOL_FN


#: Raw feature channels that vary per (person, item) cell. They are
#: concatenated to the response tensor before ``cell_proj`` and disable the
#: ``num_matrix_blocks == 0`` mean-pooling fast path (materialization cost).
_CELL_LEVEL_RAW_FEATURE_CHANNELS = ("rest_mean_score",)

#: Raw feature channels that are constant over the person axis. They are
#: computed from the raw responses and concatenated to the pooled per-item
#: representation right before the outer item stack (late fusion); they do
#: not disable the fast path.
_ITEM_LEVEL_RAW_FEATURE_CHANNELS = (
    "item_rest_correlation",
    "item_rest_correlation_shuffled",
)

_APPROVED_RAW_FEATURE_CHANNELS = (
    _CELL_LEVEL_RAW_FEATURE_CHANNELS + _ITEM_LEVEL_RAW_FEATURE_CHANNELS
)


@keras.saving.register_keras_serializable(package="bayesflow_irt")
class MatrixEquivariantIRTSummary(keras.Layer):
    """Hartford-style matrix-equivariant summary network for IRT response matrices.

    This layer maps a binary or polytomous response matrix with shape
    ``(batch, persons, items)`` or ``(batch, persons, items, categories)``
    to per-item embeddings of shape ``(batch, items, embed_dim)``.  The
    construction is:

    - person-permutation invariant: reordering respondents does not change
      the item embeddings;
    - item-permutation equivariant: reordering items reorders the output in
      the same way;
    - shape-flexible: the same trained layer can be called with different
      numbers of persons/items at inference time.

    The forward path is:

    .. code-block:: text

        (B, P, I[, F])
        -> cast non-float inputs to float32 (preserve float dtypes)
        -> optional cell-level raw channels    # concat -> (B, P, I, F+C)
        -> Dense(matrix_hidden_dim)            # cell projection
        -> Hartford EMN blocks x L             # (B, P, I, H)
        -> person pool (mean / PMA / mean+PMA) # (B, I, H')
        -> optional item-level raw channels    # concat -> (B, I, H'+M)
        -> outer item SAB/ISAB stack
        -> Dense(embed_dim)                    # (B, I, D)

    Matrix interaction is handled by Hartford-style exchangeable matrix
    blocks, person aggregation is done by either a simple mean or PMA, and
    item-context modelling is delegated to a Set Transformer style outer
    stack.  This gives a lightweight summary while preserving the
    IRT-specific symmetry constraints.

    Parameters
    ----------
    embed_dim : int
        Output width of each item embedding.
    matrix_hidden_dim : int
        Hidden width of the cell-wise representation used inside the
        matrix-equivariant blocks.
    num_matrix_blocks : int
        Number of exchangeable matrix blocks applied before person pooling.
        ``0`` is allowed and skips directly from the cell projection to
        person pooling.
    matrix_activation : str
        Activation used inside each matrix block.
    matrix_block_dropout : float or None
        **Experimental.** Dropout rate for the matrix blocks only. ``None``
        (default) reuses ``dropout``. Set this
        to vary block dropout while holding the PMA/outer-stack dropout
        fixed, so block dropout can be varied on the matrix blocks instead of
        confounding them with the rest of the network.
    matrix_block_dropout_kind : str
        **Experimental.** Dropout granularity inside the matrix blocks,
        ``"element"`` (default, shipped) or ``"channel"``. Forwarded to every
        ``_HartfordEMNBlock``; see that class's ``dropout_kind`` docstring for
        what each does and why Hartford et al. (2018) §3.2 prescribe
        ``"channel"``. Experimental.
    matrix_norm_placement : str
        **Experimental.** Where each matrix block's LayerNorm sits relative
        to the Hartford linear combination and the activation. Forwarded to
        every ``_HartfordEMNBlock``; see that class's ``norm_placement``
        docstring for the supported orderings. The default,
        ``"pre_activation"``, reproduces the standard behavior exactly.
        Experimental.
    person_pooling : str
        How to pool over the person axis after the matrix blocks.
        ``"mean"`` is the cheapest invariant reducer; ``"pma"`` uses
        pooling-by-multihead-attention for a more expressive learned reducer;
        ``"mean_pma"`` concatenates person means with PMA features;
        ``"mean_std"`` concatenates person means with per-item standard
        deviations of the *hidden* cell features, doubling the pooled width at
        no parameter cost.

        ``"mean_std"`` is not the degenerate case it may look like. Pooling the
        standard deviation of the **raw** binary responses would add nothing:
        for a Bernoulli item the person-axis variance is ``p_i (1 - p_i)``, a
        deterministic function of the column mean the mean-pooling branch
        already provides. After at least one matrix block, however, each cell
        is ``h_pi = f(x_pi, row_mean_p, col_mean_i, global)`` (Hartford et al.,
        2018), so its spread across persons depends on how that item's
        responses covary with person-level standing -- i.e. on an item-rest
        second moment, which is the classical-test-theory analog of
        discrimination (Embretson & Reise, 2000). With
        ``num_matrix_blocks=0`` the cells are a linear projection of the raw
        responses and the branch degenerates accordingly, so ``mean_std`` is
        only meaningful for ``num_matrix_blocks >= 1``.

        Using several aggregators rather than one is a standard inductive-bias
        remedy for reducers that cannot distinguish inputs a single moment
        collapses (Corso et al., 2020, on mean/std/max neighbourhood
        aggregation). Unlike ``"pma"`` this needs no attention over the person
        axis, so it does not materialize per-cell key/value projections: its
        cost is one extra reduction over a tensor the dense path already holds.
        It does not support ``person_chunk_size`` -- see the error raised in
        ``__init__`` for why.
    num_pma_heads : int
        Number of attention heads used by PMA when
        ``person_pooling`` uses PMA.
    num_pma_seeds : int
        Number of PMA seed vectors. For ``person_pooling="pma"``, pooled
        features have width ``num_pma_seeds * matrix_hidden_dim``. For
        ``person_pooling="mean_pma"``, pooled width is
        ``(1 + num_pma_seeds) * matrix_hidden_dim``.
    raw_feature_channels : tuple of str
        Optional raw response-derived feature channels. Empty by default (no
        additional channels; preserves prior behavior). Each approved channel
        has a fixed fusion point chosen by where the statistic varies and
        where it is consumed:

        - ``"rest_mean_score"`` (cell-level): each person's mean
          response/category-score over items, excluding the current item's
          own response — an ability proxy that avoids circularity with the
          item being scored. It varies per (person, item) cell, so it is
          concatenated to the response tensor before ``cell_proj`` and
          processed by the EMN stack.
        - ``"item_rest_correlation"`` (item-level): the corrected item-rest
          correlation, computed within each dataset (over the person axis,
          independently per batch element) between each item's score and its
          rest score. For binary input this is the point-biserial correlation
          and is the classical-test-theory analog of the IRT discrimination
          parameter (Embretson & Reise, 2000; DeMars, 2018) — the basis for
          including this channel at all. For one-hot polytomous input
          (``K > 2``), this is a plain Pearson correlation on
          linearly-scored (interval-treated) categories rather than a full
          polyserial/polychoric latent-variable estimator, chosen for
          implementation tractability: it is a closed-form, differentiable
          ``keras.ops`` computation, unlike polyserial/polychoric estimation.
          It is a per-item statistic consumed by the outer item encoder, so
          it is fused *late*: computed from the raw responses and
          concatenated to the pooled per-item representation right before
          the outer item stack. It is never broadcast over the person axis
          and does not widen ``cell_proj``. See the Notes section below for
          a discussion of what feeding this as an input feature does and
          does not imply.

          Its benefit should be largest when items are few (the network has
          little else from which to infer discrimination) and smallest when
          persons are few (the correlation, an average over the person
          axis, is then too noisy to carry information).
        - ``"item_rest_correlation_shuffled"`` (item-level): the same
          statistic with its values permuted across items, independently per
          dataset and redrawn every call, at evaluation time as well as during
          training. This is an **ablation control, not a feature** — it exists
          to be compared against ``"item_rest_correlation"``, and there is no
          reason to enable it in a model you intend to use. Shape, parameter
          count and per-dataset marginal distribution are identical; only the
          item-to-value pairing is destroyed. If a gain from the real channel
          survives the permutation, that gain is not information the statistic
          carries, and the honest conclusion is that something other than the
          statistic is responsible.

        ``"person_mean_score"`` and ``"item_mean_score"`` were removed: the
        first Hartford EMN block reproduces them exactly (its row/column/
        global mean terms applied to the linearly projected cells), so they
        add no information when ``num_matrix_blocks >= 1``; with zero blocks
        the mean-pooling fast-path input already *is* the per-item mean.

        For one-hot polytomous input ``(B, P, I, K)``, both channels first
        reduce each cell to a normalized ``[0, 1]`` category-score
        expectation (``sum_k onehot[..., k] * k / (K - 1)``) before computing
        rest means / correlations on that scalar. One-hot encoding of
        ``summary_variables`` happens upstream, in the adapter — this layer
        never sees the pre-encoding category index directly — but because
        one-hot encoding a hard categorical draw is lossless, the recovered
        score is numerically identical to computing directly from that index
        (just rescaled to ``[0, 1]``).

        Only *cell-level* channels disable the ``num_matrix_blocks == 0``
        mean-pooling fast path: they must be materialized per (person, item)
        cell before the projection, which defeats the fast path's
        pool-before-project memory saving. (The concatenation itself would
        still commute with person averaging — the cost is materialization,
        not algebra.) Item-level channels are computed directly from the raw
        responses and fused after pooling, so they keep the fast path
        available.
    num_outer_sabs : int
        Number of item-level Set Attention Blocks (or ISAB blocks) applied
        after person pooling.
    outer_embed_dim : int
        Hidden width of the outer item encoder.
    num_outer_heads : int
        Number of attention heads in each outer item block.
    outer_mlp_depth, outer_mlp_width : int
        Feed-forward network depth and width used inside the outer item
        blocks.
    dropout : float
        Dropout rate used in both matrix and outer item blocks.
    use_outer_isab : bool
        If ``True``, use Induced Set Attention Blocks in the outer item
        encoder to reduce attention cost for large item sets.
    num_outer_inducing_points : int
        Number of inducing points for outer ISAB blocks.
    num_parameter_heads : int
        Number of parallel projection heads applied to the shared item state
        after the outer item encoder and before the final ``output_proj``.
        ``0`` (default) preserves the current behavior exactly: ``output_proj``
        is applied directly to the outer stack's output. When ``> 0``, ``N``
        independent ``Dense(parameter_head_dim, activation="mish")`` layers are
        each applied to the *same* shared item state, concatenated, and then
        passed through ``output_proj``. This adds representational capacity
        and structure before the final projection; the heads are otherwise
        undifferentiated (identical input, identical architecture, no
        head-specific loss or routing), so they should not be read as
        specializing in any particular target parameter — nothing in the
        architecture ties a given head to a given parameter, so which head
        ends up encoding what is unconstrained and not identifiable from the
        configuration alone.

        **Experimental, not recommended for use**: kept as a tested,
        off-by-default option.
    parameter_head_dim : int or None
        Output width of each parameter head. Defaults to ``embed_dim`` when
        ``None`` (a conservative choice that keeps each head's capacity
        comparable to the final output width). Ignored when
        ``num_parameter_heads == 0``.
    item_chunk_size : int or None
        Accepted for API compatibility with the memory utilities, but
        ignored by this class.  Matrix summaries do not use an inner
        item-chunk loop, so changing this value has no effect on the
        forward pass.  A warning is emitted when a non-``None`` value is
        provided to make that explicit.
    gradient_checkpointing : bool
        If ``True`` and the backend is PyTorch, wrap the matrix-block stack in
        ``torch.utils.checkpoint.checkpoint`` during training.  This reduces
        activation memory at the cost of extra recomputation in the backward
        pass.  It is the active model-side memory knob for the **dense**
        path only: it has no effect whenever person-axis chunking is active
        (see ``person_chunk_size``), because ``call()`` routes chunked
        forward/backward through ``_chunked_matrix_and_pool``/
        ``_chunked_matrix_and_pool_trainable`` instead of
        ``_maybe_checkpointed_matrix`` -- both chunked paths already bound
        peak accelerator memory on their own. A warning is emitted at
        construction time when both are combined with
        ``num_matrix_blocks >= 1``.
    person_chunk_size : int or None
        If set and smaller than the actual number of persons, run
        ``cell_proj`` + the matrix-block stack + person-mean pooling as a
        person-chunked, CPU-staged pipeline instead of materializing the full
        ``(B, n_persons, I, matrix_hidden_dim)`` tensor on the accelerator:
        each stage's output is staged to host memory between stages, so peak
        accelerator memory scales with ``person_chunk_size``, not
        ``n_persons``. Only takes effect when
        ``num_matrix_blocks >= 1`` — the ``num_matrix_blocks == 0`` fast path
        is already ``O(n_persons)``-cheap and needs no chunking. Requires
        ``person_pooling="mean"`` (PMA-based pooling does not decompose into
        an associative per-chunk reduction) and no cell-level
        ``raw_feature_channels`` (``"rest_mean_score"``, which must be
        materialized per-cell before ``cell_proj``, ahead of this pipeline);
        both are validated at construction time. On the torch backend,
        training also uses a person-chunked, ``O(person_chunk_size)``-memory
        path: a custom ``torch.autograd.Function`` (see
        ``_get_chunked_trainable_matrix_pool_fn``) reproduces the dense
        gradients exactly by treating each block's ``col_mean``/
        ``global_mean`` (and the final person-pool) as detach points with
        their own small per-chunk backward passes, rather than staging
        through NumPy (which would detach from autograd entirely) --
        including when matrix blocks use nonzero ``dropout``, by
        snapshotting and restoring the RNG state around each chunk's
        recomputed dropout call so it reproduces the mask actually used in
        the forward pass that produced the loss. On non-torch backends,
        ``call()`` falls back to the dense, differentiable path during
        training — gradients are always correct either way, but the memory
        savings only extend to training on the torch backend. ``None``
        (default) preserves
        current dense behavior exactly.

    Notes
    -----
    ``item_chunk_size`` is serialized for compatibility with the existing
    memory-probe and retry helpers, but it is intentionally non-operational
    here.  For this class, the practical training knobs are
    ``gradient_checkpointing`` on the summary side plus batch-size / flow-loss
    controls in :mod:`bayesflow_irt.utils`.

    ``"item_rest_correlation"`` is a classical point estimate of essentially
    the same discrimination quantity the inference network's posterior over
    ``a`` is meant to estimate. This is not a validity problem for
    simulation-based training — it remains a deterministic function of the
    observed data — but it does change what this summary network does for
    that parameter: less "learn a sufficient statistic from raw responses,"
    more "calibrate uncertainty around a pre-computed point estimate."

    **Limitation — no missing-data support:** this layer assumes **complete
    (non-missing) response matrices**. There is no mask argument anywhere in
    ``call()``/``build()``; person-axis pooling (mean, PMA, or mean+PMA)
    always averages/attends over every person slot in the input, including
    the Rasch/PCM "fast path" (``num_matrix_blocks == 0`` and
    ``person_pooling == "mean"``). If you pad a batch to a rectangular shape
    (e.g. for adaptive testing or planned-missingness designs), the padding
    sentinel is pooled in as if it were a real response, biasing the
    resulting item/person embeddings. There is currently no way to exclude
    padded entries from pooling. If your data has genuine missingness,
    either avoid padding (e.g. simulate/train only at fixed sizes) or treat
    this as an open limitation before deploying on real, possibly-incomplete
    response data.

    References
    ----------
    .. [1] Hartford, J. et al. (2018). *Deep Models of Interactions Across
       Sets*. ICML 2018.
    .. [2] Maron, H. et al. (2019). *Provably Powerful Graph Networks*.
       NeurIPS 2019.
    .. [3] Lee, J. et al. (2019). *Set Transformer*. ICML 2019.
    .. [4] Zaheer, M. et al. (2017). *Deep Sets*. NeurIPS 2017.
    .. [5] Embretson, S. E., & Reise, S. P. (2000). *Item Response Theory for
       Psychologists*. Lawrence Erlbaum Associates. Basis for
       ``"item_rest_correlation"``: the biserial correlation of an item with
       total score is directly related to the IRT discrimination parameter.
    .. [6] DeMars, C. E. (2018). Classical Test Theory and Item Response
       Theory. In P. Irwing, T. Booth, & D. J. Hughes (Eds.), *The Wiley
       Handbook of Psychometric Testing* (pp. 49-73). John Wiley & Sons.
       https://doi.org/10.1002/9781118489772.ch2. Basis for using the
       corrected item-*rest* (not item-total) correlation, which excludes an
       item's own response to avoid part-whole correlation inflation.
    """

    supports_item_chunking = False
    supports_person_chunking = True

    def __init__(
        self,
        embed_dim: int = 32,
        *,
        matrix_hidden_dim: int = 32,
        num_matrix_blocks: int = 2,
        matrix_activation: str = "mish",
        matrix_block_dropout: float | None = None,
        matrix_block_dropout_kind: str = "element",
        matrix_norm_placement: str = "pre_activation",
        person_pooling: str = "mean",
        num_pma_heads: int = 4,
        num_pma_seeds: int = 1,
        raw_feature_channels: tuple[str, ...] = (),
        num_outer_sabs: int = 2,
        outer_embed_dim: int = 64,
        num_outer_heads: int = 4,
        outer_mlp_depth: int = 2,
        outer_mlp_width: int = 128,
        dropout: float = 0.05,
        use_outer_isab: bool = False,
        num_outer_inducing_points: int = 32,
        num_parameter_heads: int = 0,
        parameter_head_dim: int | None = None,
        item_chunk_size: int | None = None,
        gradient_checkpointing: bool = False,
        person_chunk_size: int | None = None,
        **kwargs,
    ):
        super().__init__(**kwargs)

        if matrix_hidden_dim <= 0:
            raise ValueError("matrix_hidden_dim must be a positive integer.")
        if num_matrix_blocks < 0:
            raise ValueError("num_matrix_blocks must be >= 0.")
        if person_pooling not in (
            "mean", "pma", "mean_pma", "mean_std", "mean_max"
        ):
            raise ValueError(
                "person_pooling must be 'mean', 'pma', 'mean_pma', "
                "'mean_std', or 'mean_max'."
            )
        if num_pma_heads <= 0:
            raise ValueError("num_pma_heads must be a positive integer.")
        if num_pma_seeds <= 0:
            raise ValueError("num_pma_seeds must be a positive integer.")
        raw_feature_channels = tuple(raw_feature_channels)
        invalid_channels = set(raw_feature_channels) - set(
            _APPROVED_RAW_FEATURE_CHANNELS
        )
        if invalid_channels:
            raise ValueError(
                f"Invalid raw_feature_channels {sorted(invalid_channels)}. "
                f"Approved channels are {_APPROVED_RAW_FEATURE_CHANNELS}. "
                "('person_mean_score' and 'item_mean_score' were removed: the "
                "Hartford EMN basis computes person/item means internally.)"
            )
        if num_outer_inducing_points <= 0:
            raise ValueError("num_outer_inducing_points must be a positive integer.")
        if num_parameter_heads < 0:
            raise ValueError("num_parameter_heads must be >= 0.")
        if parameter_head_dim is not None and parameter_head_dim <= 0:
            raise ValueError("parameter_head_dim must be a positive integer or None.")
        if item_chunk_size is not None and item_chunk_size <= 0:
            raise ValueError("item_chunk_size must be a positive integer or None.")
        if item_chunk_size is not None:
            warnings.warn(
                "MatrixEquivariantIRTSummary ignores item_chunk_size because it does "
                "not use item-axis chunking. The argument is kept for API "
                "compatibility.",
                stacklevel=2,
            )
        if person_chunk_size is not None and person_chunk_size <= 0:
            raise ValueError("person_chunk_size must be a positive integer or None.")
        if person_chunk_size is not None and person_pooling != "mean":
            raise ValueError(
                "person_chunk_size requires person_pooling='mean'. PMA-based "
                "pooling does not decompose into an associative per-chunk "
                "reduction at all -- attention over persons needs every "
                "person in scope at once -- so that half of this restriction "
                "is mathematical. 'mean_std' and 'mean_max' *do* decompose "
                "(pairwise-merged mean/M2 in the first case, a running max in "
                "the second) and are blocked only by the implementation: the "
                "chunked paths stage a single per-chunk person sum, and the "
                "backward pass drops the last block's staged output because a "
                "mean's gradient does not depend on it. Both std and max need "
                "those values back to route their gradients. Use the dense "
                "path meanwhile: it costs one extra reduction over a tensor "
                "the dense path already materializes."
            )
        if person_chunk_size is not None and any(
            c in _CELL_LEVEL_RAW_FEATURE_CHANNELS for c in raw_feature_channels
        ):
            raise ValueError(
                "person_chunk_size is not yet supported together with "
                f"cell-level raw_feature_channels {_CELL_LEVEL_RAW_FEATURE_CHANNELS}: "
                "they must be materialized per-cell before cell_proj, ahead of "
                "the chunked pipeline."
            )
        if person_chunk_size is not None and matrix_block_dropout_kind in (
            "channel",
            "row_col",
        ):
            raise ValueError(
                "person_chunk_size is not supported together with "
                f"matrix_block_dropout_kind={matrix_block_dropout_kind!r}: its mask "
                "is shared across the person and/or item axes and so must be drawn "
                "once for the whole matrix, but each chunk's dropout call draws "
                "its own."
            )
        if (
            person_chunk_size is not None
            and matrix_norm_placement == "residual_pre_norm"
        ):
            raise ValueError(
                "person_chunk_size is not supported together with "
                "matrix_norm_placement='residual_pre_norm': the norm is applied "
                "to the block input, changing the tensor the chunked col_mean/"
                "global_mean reductions accumulate over."
            )
        if (
            person_chunk_size is not None
            and gradient_checkpointing
            and num_matrix_blocks > 0
        ):
            warnings.warn(
                "gradient_checkpointing has no effect whenever person-axis "
                "chunking is active: once person_chunk_size is smaller than "
                "the actual n_persons at call time, call() routes through "
                "_chunked_matrix_and_pool_trainable (training) or "
                "_chunked_matrix_and_pool (inference), neither of which goes "
                "through _maybe_checkpointed_matrix -- both chunked paths "
                "already bound peak accelerator memory to "
                "O(person_chunk_size) on their own. gradient_checkpointing "
                "only affects the dense fallback path (person_chunk_size is "
                "None, person_chunk_size >= n_persons at call time, or the "
                "training-time chunking gate is unavailable, i.e. a "
                "non-torch backend).",
                stacklevel=2,
            )

        self.item_chunk_size = item_chunk_size
        self.gradient_checkpointing = gradient_checkpointing
        self.person_chunk_size = person_chunk_size
        self._backend_is_torch = keras.backend.backend() == "torch"

        self._config = dict(
            embed_dim=embed_dim,
            matrix_hidden_dim=matrix_hidden_dim,
            num_matrix_blocks=num_matrix_blocks,
            matrix_activation=matrix_activation,
            matrix_block_dropout=matrix_block_dropout,
            matrix_block_dropout_kind=matrix_block_dropout_kind,
            matrix_norm_placement=matrix_norm_placement,
            person_pooling=person_pooling,
            num_pma_heads=num_pma_heads,
            num_pma_seeds=num_pma_seeds,
            raw_feature_channels=raw_feature_channels,
            num_outer_sabs=num_outer_sabs,
            outer_embed_dim=outer_embed_dim,
            num_outer_heads=num_outer_heads,
            outer_mlp_depth=outer_mlp_depth,
            outer_mlp_width=outer_mlp_width,
            dropout=dropout,
            use_outer_isab=use_outer_isab,
            num_outer_inducing_points=num_outer_inducing_points,
            num_parameter_heads=num_parameter_heads,
            parameter_head_dim=parameter_head_dim,
            item_chunk_size=item_chunk_size,
            gradient_checkpointing=gradient_checkpointing,
            person_chunk_size=person_chunk_size,
        )

        self.person_pooling = person_pooling
        self.num_pma_seeds = num_pma_seeds
        self.raw_feature_channels = raw_feature_channels
        self._cell_level_channels = tuple(
            c for c in raw_feature_channels if c in _CELL_LEVEL_RAW_FEATURE_CHANNELS
        )
        self._item_level_channels = tuple(
            c for c in raw_feature_channels if c in _ITEM_LEVEL_RAW_FEATURE_CHANNELS
        )
        self.cell_proj = layers.Dense(
            matrix_hidden_dim, activation=None, name="cell_proj"
        )
        block_dropout = (
            dropout if matrix_block_dropout is None else float(matrix_block_dropout)
        )
        self.matrix_blocks = [
            _HartfordEMNBlock(
                matrix_hidden_dim,
                activation=matrix_activation,
                dropout=block_dropout,
                dropout_kind=matrix_block_dropout_kind,
                norm_placement=matrix_norm_placement,
                name=f"matrix_block_{idx}",
            )
            for idx in range(num_matrix_blocks)
        ]

        if person_pooling in ("pma", "mean_pma"):
            self.person_pma = PoolingByMultiHeadAttention(
                num_seeds=num_pma_seeds,
                embed_dim=matrix_hidden_dim,
                num_heads=num_pma_heads,
                dropout=dropout,
            )
        else:
            self.person_pma = None

        if person_pooling == "mean":
            self._pooled_dim = matrix_hidden_dim
        elif person_pooling == "pma":
            self._pooled_dim = num_pma_seeds * matrix_hidden_dim
        elif person_pooling in ("mean_std", "mean_max"):
            self._pooled_dim = 2 * matrix_hidden_dim
        else:
            self._pooled_dim = (1 + num_pma_seeds) * matrix_hidden_dim

        # Item-level raw channels are concatenated to the pooled per-item
        # representation, so the outer stack's input is wider than the pooled
        # width whenever such channels are enabled.
        self._outer_input_dim = self._pooled_dim + len(self._item_level_channels)

        self.outer_sabs, self.bridge, self.output_proj = _make_outer_item_layers(
            input_embed_dim=self._outer_input_dim,
            embed_dim=embed_dim,
            num_outer_sabs=num_outer_sabs,
            outer_embed_dim=outer_embed_dim,
            num_outer_heads=num_outer_heads,
            outer_mlp_depth=outer_mlp_depth,
            outer_mlp_width=outer_mlp_width,
            dropout=dropout,
            use_outer_isab=use_outer_isab,
            num_outer_inducing_points=num_outer_inducing_points,
            layer_prefix="matrix_",
        )

        self.num_parameter_heads = num_parameter_heads
        self._parameter_head_dim = (
            embed_dim if parameter_head_dim is None else parameter_head_dim
        )
        if num_parameter_heads > 0:
            self.parameter_head_layers = [
                layers.Dense(
                    self._parameter_head_dim,
                    activation="mish",
                    name=f"matrix_head_{idx}",
                )
                for idx in range(num_parameter_heads)
            ]
            self._output_proj_input_dim = num_parameter_heads * self._parameter_head_dim
        else:
            self.parameter_head_layers = None
            self._output_proj_input_dim = outer_embed_dim

    def build(self, input_shape):
        if len(input_shape) not in (3, 4):
            raise ValueError(
                f"Expected 3D (batch, P, I) or 4D (batch, P, I, K) input, "
                f"got {len(input_shape)}D shape {input_shape}"
            )

        if len(input_shape) == 4:
            batch, num_persons, num_items, feature_dim = input_shape
        else:
            batch, num_persons, num_items = input_shape
            feature_dim = 1

        matrix_hidden_dim = self._config["matrix_hidden_dim"]
        outer_embed_dim = self._config["outer_embed_dim"]

        augmented_feature_dim = feature_dim + len(self._cell_level_channels)
        self.cell_proj.build((None, num_persons, num_items, augmented_feature_dim))
        # Weight shapes for cell_proj/matrix_blocks depend only on the
        # channel dim (F/H), never on the person count -- Dense layers size
        # weights from the last axis, and the Hartford EMN blocks' row/col/
        # global projections build with `None` in the person/item axes (see
        # _HartfordEMNBlock.build). So this dummy forward pass, used purely
        # to trigger weight initialization, can use a small constant person
        # count instead of the real (possibly huge, e.g. 1,000,000) N. Using
        # the real N here would materialize an O(N x I x H) tensor through
        # cell_proj even when num_matrix_blocks == 0, defeating the whole
        # point of the mean-pooling fast path in call() (see there).
        dummy_num_persons = min(num_persons, 2) if num_persons is not None else 2
        dummy = ops.zeros((2, dummy_num_persons, num_items, feature_dim))
        dummy = self._augment_with_cell_features(dummy)
        dummy = self.cell_proj(dummy)
        for block in self.matrix_blocks:
            if not block.built:
                dummy = block(dummy, training=False)

        if self.person_pma is not None and not self.person_pma.built:
            # BayesFlow 2.0.10 PMA has no build() implementation. An eager
            # forward creates its nested attention/MLP weights before Keras
            # restores the checkpoint, rather than marking unbuilt state built.
            # Its weights do not depend on the person count, so the same small
            # dummy count as above keeps a symbolic person axis buildable.
            self.person_pma(
                ops.zeros((2, dummy_num_persons, matrix_hidden_dim)),
                training=False,
            )

        _build_outer_item_layers(
            outer_sabs=self.outer_sabs,
            bridge=self.bridge,
            output_proj=self.output_proj,
            num_items=num_items,
            input_embed_dim=self._outer_input_dim,
            outer_embed_dim=outer_embed_dim,
            output_proj_input_dim=self._output_proj_input_dim,
        )
        if self.parameter_head_layers is not None:
            for head_layer in self.parameter_head_layers:
                if not head_layer.built:
                    head_layer.build((None, num_items, outer_embed_dim))

        super().build(input_shape)

    def _matrix_forward(self, x, training):
        for block in self.matrix_blocks:
            x = block(x, training=training)
        return x

    def _maybe_checkpointed_matrix(self, x, training):
        """Optionally wrap ``_matrix_forward`` with gradient checkpointing.

        See :func:`_checkpoint_if_enabled` for the exact conditions.
        """
        return _checkpoint_if_enabled(
            self._matrix_forward,
            x,
            training,
            gradient_checkpointing=self.gradient_checkpointing,
            backend_is_torch=self._backend_is_torch,
            extra_guard=len(self.matrix_blocks) > 0,
        )

    def _chunked_matrix_and_pool(self, x, training):
        """Person-chunked, CPU-staged ``cell_proj`` + matrix blocks + mean pool.

        Implements between-block staging: ``x`` is processed
        in ``person_chunk_size``-sized person-axis slices, and every stage's
        full output (``cell_proj``'s, then each ``_HartfordEMNBlock``'s) is
        staged to host memory via ``ops.convert_to_numpy`` between stages —
        the full ``(B, n_persons, I, H)`` tensor is never concatenated back
        on the accelerator, so peak accelerator memory is
        ``O(person_chunk_size)``, not ``O(n_persons)``. The final person-mean
        pool is itself an associative reduction (like ``col_mean``/
        ``global_mean`` inside each block), so it is computed the same way —
        chunked sums accumulated while streaming the last stage's staged
        chunks — rather than gathering everything back first.

        **Inference-only — breaks autograd, not just unbounded backward
        memory.** ``ops.convert_to_numpy`` detaches from the computation
        graph (unlike ``_HartfordEMNBlock._chunked_call``, which
        stays in pure tensor ops and is therefore already differentiable).
        This is now the *fallback* path: ``call()`` uses this whenever
        ``training=False``, or whenever ``training=True`` but the
        gradient-aware trainable path (``_chunked_matrix_and_pool_trainable``)
        isn't available (non-torch backend) — see that method for the path
        that does extend the memory savings to training.
        """
        n_persons = x.shape[1]
        chunk_size = self.person_chunk_size
        starts = list(range(0, n_persons, chunk_size))

        # cell_proj is a plain per-cell Dense (no cross-person reduction), so
        # it chunks trivially: project each chunk, stage immediately.
        staged = [
            ops.convert_to_numpy(
                self.cell_proj(x[:, s : min(s + chunk_size, n_persons)])
            )
            for s in starts
        ]

        for block in self.matrix_blocks:
            n_items = staged[0].shape[2]
            col_term, global_term = block._stats_terms(
                (ops.convert_to_tensor(arr) for arr in staged),
                n_persons=n_persons,
                n_items=n_items,
            )
            staged = [
                ops.convert_to_numpy(
                    block._apply_chunk(
                        ops.convert_to_tensor(arr), col_term, global_term, training
                    )
                )
                for arr in staged
            ]

        return _widened_person_mean(staged, n_persons)

    def _person_chunking_config_is_eligible(self) -> bool:
        """Whether this instance's config allows person-axis chunking at all.

        ``__init__`` validates ``person_pooling == "mean"`` and no
        cell-level ``raw_feature_channels`` -- but only when
        ``person_chunk_size`` is provided *at construction time*.
        ``person_chunk_size`` is a plain mutable attribute that can be
        changed afterward (e.g. by
        ``bayesflow_irt.utils.set_summary_memory_config`` during
        ``fit_with_oom_retry``/``probe_memory_config``), which would bypass
        that one-time check. ``call()`` and
        ``bayesflow_irt.utils._supports_person_chunking`` both re-derive
        eligibility from this method rather than trusting that a non-``None``
        ``person_chunk_size`` implies a valid configuration -- otherwise a
        PMA-pooling or cell-level-raw-feature instance could have chunking
        silently (and incorrectly) enabled post-construction, skipping PMA
        pooling or the required cell-feature augmentation entirely.
        """
        return self.person_pooling == "mean" and not self._cell_level_channels

    def _can_use_trainable_chunking(self, x):
        """Whether ``_chunked_matrix_and_pool_trainable`` is safe to use.

        Requires the torch backend -- the mechanism is a
        ``torch.autograd.Function``.

        Nonzero dropout on a matrix block no longer disables this path *on
        CPU and CUDA*: the backward pass recomputes each chunk's apply pass
        and would draw a different dropout mask than the one that produced
        the pooled output used in the loss unless the RNG state is restored
        first -- which ``_get_chunked_trainable_matrix_pool_fn`` does by
        snapshotting/restoring ``torch``'s CPU and CUDA RNG state (see its
        docstring).

        That snapshot only covers CPU and CUDA. A torch device with its own
        RNG that is neither -- notably Apple MPS -- consumes a stream this
        path does not capture, so backward would recompute dropout with a
        *different* mask than forward used, silently computing gradients for
        a different realized forward pass. When any matrix block has nonzero
        dropout, restrict the trainable path to CPU/CUDA and let other
        devices fall back to the dense path. With no dropout anywhere there
        is no stochastic op to reproduce, so every torch device is safe.
        """
        if not self._backend_is_torch:
            return False
        if not any(block.dropout > 0 for block in self.matrix_blocks):
            return True
        device = getattr(x, "device", None)
        device_type = getattr(device, "type", None)
        return device_type in ("cpu", "cuda")

    def _chunked_matrix_and_pool_trainable(self, x, training):
        """Gradient-correct, memory-bounded person-chunked forward.

        Thin wrapper around the ``torch.autograd.Function`` built by
        ``_get_chunked_trainable_matrix_pool_fn`` — see that function's
        docstring for the full design.
        Unlike ``_chunked_matrix_and_pool``, this extends the
        ``O(person_chunk_size)`` memory bound to training: the returned
        tensor is a genuine, differentiable node in the outer autograd
        graph, so callers do not need a ``not training`` guard around this
        path the way they do around the NumPy-staged fallback.
        """
        chunked_fn = _get_chunked_trainable_matrix_pool_fn()
        # torch.autograd.Function only wires up a backward node when at
        # least one tensor input requires grad; x (raw response data) never
        # does on its own. Mirrors _checkpoint_if_enabled's identical
        # workaround (networks.py, _checkpoint_if_enabled). Captured before
        # the workaround forces requires_grad_(True), so the Function's
        # backward can tell "genuinely needs its gradient" (some upstream
        # trainable layer feeds x) apart from "only forced True to hook
        # this Function's output into the graph" -- see the Function
        # docstring's x_needs_grad note.
        x_needs_grad = x.requires_grad
        if not x_needs_grad:
            x = x.detach().requires_grad_(True)
        return chunked_fn.apply(x, self, self.person_chunk_size, training, x_needs_grad)

    def _cell_score(self, x):
        """Reduce a ``(B, P, I, F)`` response tensor to a scalar ``(B, P, I)`` score.

        For binary/pre-scored input (``F == 1``), the score is the response
        value itself. For one-hot polytomous input (``F == K``), the score is
        a normalized ``[0, 1]`` category-score expectation
        (``sum_k onehot[..., k] * k / (K - 1)``), keeping the score on a
        comparable scale across items/models with different numbers of
        response categories.
        """
        num_categories = x.shape[-1]
        if num_categories == 1:
            return x[..., 0]
        category_scores = ops.cast(ops.arange(num_categories), x.dtype) / ops.cast(
            max(num_categories - 1, 1), x.dtype
        )
        return ops.sum(x * category_scores, axis=-1)

    def _rest_score(self, score):
        """Per-person mean score excluding the current item's own response.

        ``score`` is ``(B, P, I)``; returns ``(B, P, I)``.
        """
        item_total = ops.sum(score, axis=2, keepdims=True)  # (B, P, 1)
        # Keep the count in tensor operations: converting a symbolic shape
        # through maximum/cast can specialize the item axis during ONNX export.
        # Only one person slice is needed for counting; broadcast over persons.
        item_count = ops.sum(ops.ones_like(score[:, :1, :]), axis=2, keepdims=True)
        denom = ops.maximum(item_count - 1, ops.ones_like(item_count))
        return (item_total - score) / denom

    def _item_rest_correlation(self, score, rest, eps=1e-6):
        """Corrected item-rest point-biserial correlation, per item.

        ``score`` and ``rest`` are ``(B, P, I)``; the correlation is computed
        within each dataset — over the person axis, independently per batch
        element and item — returning ``(B, 1, I)``. The ``eps``-guarded
        denominator makes a zero-variance item (e.g. everyone right or
        everyone wrong within a dataset) resolve to ``0`` rather than
        ``NaN``: with zero variance, the covariance term is also exactly
        ``0``, so the ratio is ``0 / eps == 0``.
        """
        mean_score = ops.mean(score, axis=1, keepdims=True)
        mean_rest = ops.mean(rest, axis=1, keepdims=True)
        d_score = score - mean_score
        d_rest = rest - mean_rest
        cov = ops.mean(d_score * d_rest, axis=1, keepdims=True)
        std_score = ops.sqrt(ops.mean(ops.square(d_score), axis=1, keepdims=True))
        std_rest = ops.sqrt(ops.mean(ops.square(d_rest), axis=1, keepdims=True))
        return cov / (std_score * std_rest + eps)

    def _augment_with_cell_features(self, x):
        """Concatenate cell-level raw feature channels before ``cell_proj``.

        ``x`` is the cast ``(B, P, I, F)`` response tensor. Returns ``x``
        unchanged when no cell-level channel is enabled (default).
        """
        if not self._cell_level_channels:
            return x

        score = self._cell_score(x)  # (B, P, I)
        rest = self._rest_score(score)  # (B, P, I)
        return ops.concatenate([x, ops.expand_dims(rest, axis=-1)], axis=-1)

    def _compute_item_features(self, x):
        """Compute item-level raw feature channels for late fusion.

        ``x`` is the cast ``(B, P, I, F)`` response tensor. Returns a
        ``(B, I, n_channels)`` tensor, or ``None`` when no item-level channel
        is enabled (default). The result is concatenated to the pooled
        per-item representation right before the outer item stack — never
        broadcast over the person axis.
        """
        if not self._item_level_channels:
            return None

        score = self._cell_score(x)  # (B, P, I)
        rest = self._rest_score(score)  # (B, P, I)
        corr = self._item_rest_correlation(score, rest)  # (B, 1, I)
        corr = ops.transpose(corr, axes=(0, 2, 1))  # (B, I, 1)

        features = [
            corr
            if channel == "item_rest_correlation"
            else self._shuffle_over_items(corr)
            for channel in self._item_level_channels
        ]
        return ops.concatenate(features, axis=-1)  # (B, I, n_channels)

    def _shuffle_over_items(self, features):
        """Permute an item-level channel across items, per dataset.

        The ablation control for ``"item_rest_correlation"``: identical shape,
        identical parameter count, identical marginal distribution within each
        dataset -- only the item-to-value pairing is destroyed. A gain that
        survives this permutation is therefore not coming from the statistic's
        information content, but from the extra channel itself (its scale, or
        the conditioning it happens to give the outer stack).

        The permutation is redrawn on every call and applied at evaluation time
        as well as during training. Shuffling only while training would hand
        the network the intact channel at evaluation and measure nothing.
        """
        noise = keras.random.uniform(ops.shape(features))
        order = ops.argsort(noise, axis=1)
        return ops.take_along_axis(features, order, axis=1)

    def _pma_pool_over_persons(self, x, training):
        batch_size = ops.shape(x)[0]
        num_persons = ops.shape(x)[1]
        num_items = ops.shape(x)[2]
        x = ops.transpose(x, axes=(0, 2, 1, 3))
        x = ops.reshape(x, (batch_size * num_items, num_persons, -1))
        x = self.person_pma(x, training=training)
        return ops.reshape(x, (batch_size, num_items, -1))

    #: Floor inside the ``mean_std`` standard deviation. ``d/dv sqrt(v)`` is
    #: unbounded as ``v -> 0``, and the variance *is* exactly zero whenever a
    #: hidden channel is constant across persons -- a dead unit, or any item
    #: answered identically by everyone. Without the floor those cells emit
    #: ``inf``/``nan`` gradients that poison the whole update.
    _STD_POOL_EPS = 1e-6

    def _pool_over_persons(self, x, training):
        if self.person_pooling == "mean":
            return ops.mean(x, axis=1)
        if self.person_pooling == "mean_std":
            mean = ops.mean(x, axis=1)
            # Deviations from the per-item mean, then a mean of squares --
            # rather than E[x^2] - E[x]^2, which cancels catastrophically when
            # the mean is large relative to the spread.
            centered = x - ops.expand_dims(mean, axis=1)
            var = ops.mean(ops.square(centered), axis=1)
            std = ops.sqrt(var + self._STD_POOL_EPS)
            return ops.concatenate((mean, std), axis=-1)
        if self.person_pooling == "mean_max":
            # Max is the third classical permutation-invariant aggregator
            # alongside sum/mean, and the one that is not a summary of the
            # bulk: it is driven by the extreme person for each item, so it
            # can separate inputs the mean collapses (Corso et al., 2020).
            # Appended to the mean rather than replacing it, so the option is
            # a strict superset of the default.
            mean = ops.mean(x, axis=1)
            mx = ops.max(x, axis=1)
            return ops.concatenate((mean, mx), axis=-1)
        pma = self._pma_pool_over_persons(x, training=training)
        if self.person_pooling == "pma":
            return pma
        mean = ops.mean(x, axis=1)
        return ops.concatenate((mean, pma), axis=-1)

    def call(self, x, training=False, **kwargs):
        ndim = len(ops.shape(x))
        if ndim not in (3, 4):
            raise ValueError(
                f"Expected 3D (batch, P, I) or 4D (batch, P, I, K) input, "
                f"got {ndim}D tensor with shape {ops.shape(x)}"
            )
        if ndim == 3:
            x = ops.expand_dims(x, axis=-1)

        x_dtype = keras.backend.standardize_dtype(x.dtype)
        if x_dtype not in ("float16", "bfloat16", "float32", "float64"):
            # Only cast integer/bool inputs to float; preserve existing float
            # dtypes (including float16/bfloat16 for mixed-precision policies).
            target_dtype = keras.backend.standardize_dtype(
                self.compute_dtype or keras.backend.floatx()
            )
            x = ops.cast(x, target_dtype)

        # Item-level raw channels are computed from the raw responses and
        # fused after person pooling, so they are orthogonal to the pooling
        # order and compatible with the fast path below.
        item_features = self._compute_item_features(x)  # (B, I, n) or None

        # Fast path for Rasch/PCM-style settings: when there are no matrix
        # blocks and pooling is mean, the linear cell projection commutes
        # with person averaging (mean(Dense(x)) == Dense(mean(x))), so mean
        # is taken *before* cell_proj here, on the smaller (B, P, I, F)
        # tensor, instead of after it on the larger (B, P, I, H) tensor the
        # general path below produces. This is an intentional memory/compute
        # optimization exploiting linearity, not a duplicate reimplementation
        # of _pool_over_persons — folding it into the general path would lose
        # the optimization unless this commutation is preserved as an
        # explicit special case. Only cell-level raw channels disable it:
        # they must be materialized per (person, item) cell before the
        # projection, which defeats the pool-before-project memory saving.
        n_persons = x.shape[1]
        use_person_chunking = (
            self.person_chunk_size is not None
            and n_persons is not None
            and self.person_chunk_size < n_persons
            and len(self.matrix_blocks) > 0
            and self._person_chunking_config_is_eligible()
        )
        # During training, the NumPy-staged path (_chunked_matrix_and_pool)
        # detaches from the autograd graph, so it is only safe when a
        # gradient-aware alternative isn't available (see
        # _chunked_matrix_and_pool_trainable's docstring) -- otherwise
        # fall back to the dense, differentiable path during training so
        # gradients are never silently broken.
        use_trainable_chunking = (
            use_person_chunking and training and self._can_use_trainable_chunking(x)
        )
        # Inference always uses the (simpler) NumPy-staged path. It must
        # never run during training (it detaches from autograd) -- when
        # training=True and use_trainable_chunking is False, this falls
        # through to the dense `else` branch below instead.
        use_numpy_staged_chunking = use_person_chunking and not training

        if (
            len(self.matrix_blocks) == 0
            and self.person_pooling == "mean"
            and not self._cell_level_channels
        ):
            x = ops.mean(x, axis=1)  # (B, I, F)
            x = self.cell_proj(x)  # (B, I, H)
        elif use_trainable_chunking:
            # use_person_chunking's _person_chunking_config_is_eligible()
            # check guarantees person_pooling == "mean" and no cell-level
            # raw channels here, regardless of whether that was true only at
            # __init__ time or person_chunk_size was mutated afterward.
            x = self._chunked_matrix_and_pool_trainable(x, training=training)
        elif use_numpy_staged_chunking:
            x = self._chunked_matrix_and_pool(x, training=training)  # (B, I, H)
        else:
            x = self._augment_with_cell_features(x)  # (B, P, I, F + C)
            x = self.cell_proj(x)  # (B, P, I, H)
            x = self._maybe_checkpointed_matrix(x, training=training)
            x = self._pool_over_persons(x, training=training)

        if item_features is not None:
            x = ops.concatenate((x, item_features), axis=-1)  # (B, I, H' + M)

        x = _apply_outer_bridge_and_sabs(
            x, outer_sabs=self.outer_sabs, bridge=self.bridge, training=training
        )
        if self.parameter_head_layers:
            # Parallel heads share identical input and architecture; they are
            # undifferentiated capacity, not routed per target parameter (see
            # the num_parameter_heads docstring).
            x = ops.concatenate(
                [head(x, training=training) for head in self.parameter_head_layers],
                axis=-1,
            )
        return self.output_proj(x)

    def get_config(self):
        base = super().get_config()
        return base | self._config

    def compute_output_shape(self, input_shape):
        batch = input_shape[0]
        num_items = input_shape[2]
        return (batch, num_items, self._config["embed_dim"])


# Checkpoints written before this class was registered carry the bare class name
# as their ``registered_name``, which the Keras registry (keyed only by
# ``"bayesflow_irt>Name"``) would not resolve. Register the bare name as an alias
# so those checkpoints keep loading without an explicit ``custom_objects`` dict.
keras.saving.get_custom_objects()["MatrixEquivariantIRTSummary"] = (
    MatrixEquivariantIRTSummary
)
