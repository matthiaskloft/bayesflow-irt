# References

Verified via OpenAlex.

## Neural set and matrix architectures

Hartford, J., Graham, D., Leyton-Brown, K., & Ravanbakhsh, S. (2018). *Deep models of interactions across sets*. In *Proceedings of the 35th International Conference on Machine Learning* (PMLR 80, pp. 1909-1918). https://proceedings.mlr.press/v80/hartford18a.html
OpenAlex verification: `https://openalex.org/W2793523712` (arXiv DOI record `10.48550/arXiv.1803.02879`).
Sec. 3.2 identifies its channel dropout with *SpatialDropout* (Tompson et al., 2015, below); the two are equivalent.
Reference implementations, cited in the paper's footnote 2: TensorFlow, https://github.com/mravanba/deep_exchangeable_tensors; PyTorch, https://github.com/jhartford/AutoEncSets. Neither contains any normalization layer, both apply the activation directly to the linear combination, and the TF version adds a post-activation skip connection when input and output channel counts match. Neither implements the whole-channel dropout Sec. 3.2 describes: the function named `channel_dropout_sparse` is element-wise, `matrix_dropout` drops rows and columns, and the PyTorch training script uses plain element-wise `nn.Dropout`. These findings back the `residual_no_norm` norm placement and the `row_col` dropout kind in `_HartfordEMNBlock`.

Tompson, J., Goroshin, R., Jain, A., LeCun, Y., & Bregler, C. (2015). *Efficient object localization using convolutional networks*. In *2015 IEEE Conference on Computer Vision and Pattern Recognition* (pp. 648-656). https://doi.org/10.1109/CVPR.2015.7298664
OpenAlex verification: `https://openalex.org/W1936750108`.
Origin of *SpatialDropout* (Sec. 3.2), the channel-dropout scheme Hartford et al. (2018, Sec. 3.2) prescribe for exchangeable matrix layers. Backs `_HartfordEMNBlock`'s `dropout_kind="channel"` option.

Corso, G., Cavalleri, L., Beaini, D., Liò, P., & Veličković, P. (2020). *Principal neighbourhood aggregation for graph nets*. In *Advances in Neural Information Processing Systems 33* (pp. 13260-13271). https://doi.org/10.48550/arXiv.2004.05718
OpenAlex verification: `https://openalex.org/works?search=Principal+Neighbourhood+Aggregation+for+Graph+Nets` (arXiv DOI record `10.48550/arXiv.2004.05718`; NeurIPS 2020 version has no DOI).
Backs `MatrixEquivariantIRTSummary`'s `person_pooling="mean_std"`: multiple aggregators (mean together with standard deviation) rather than a single one, as a remedy for reducers that cannot distinguish inputs a single moment collapses. Cited for the multi-aggregator design choice only -- the paper concerns message passing over graph neighbourhoods, not exchangeable matrices, and makes no claim about item response models.

Maron, H., Ben-Hamu, H., Serviansky, H., & Lipman, Y. (2019). *Provably powerful graph networks*. In *Advances in Neural Information Processing Systems* (NeurIPS 2019).
OpenAlex verification: `https://openalex.org/W2945568301` (arXiv DOI record `10.48550/arXiv.1905.11136`).

Zaheer, M., Kottur, S., Ravanbakhsh, S., Poczos, B., Salakhutdinov, R., & Smola, A. J. (2017). *Deep sets*. In *Advances in Neural Information Processing Systems* (Vol. 30, pp. 3391-3401).
OpenAlex verification: `https://openalex.org/W2753798143`, `https://openalex.org/W2952433032`.

Lee, J., Lee, Y., Kim, J., Kosiorek, A. R., Choi, S., & Teh, Y. W. (2019). *Set Transformer: A framework for attention-based permutation-invariant neural networks*. In *Proceedings of the 36th International Conference on Machine Learning* (PMLR 97, pp. 3744-3753). https://proceedings.mlr.press/v97/lee19d.html
OpenAlex verification: `https://openalex.org/W2894384847` (arXiv DOI record `10.48550/arXiv.1810.00825`).

## IRT foundations and diagnostics

Birnbaum, A. (1968). Some latent trait models and their use in inferring an examinee's ability. In F. M. Lord & M. R. Novick (Eds.), *Statistical theories of mental test scores* (pp. 395-479). Addison-Wesley.
OpenAlex verification: `https://openalex.org/W1547716543`.

Masters, G. N. (1982). A Rasch model for partial credit scoring. *Psychometrika, 47*(2), 149-174. https://doi.org/10.1007/BF02296272
OpenAlex verification: `https://openalex.org/W1966576448`.

Muraki, E. (1992). A generalized partial credit model: Application of an EM algorithm. *Applied Psychological Measurement, 16*(2), 159-176. https://doi.org/10.1177/014662169201600206
OpenAlex verification: `https://openalex.org/W2160900637`.

Chalmers, R. P. (2012). mirt: A multidimensional item response theory package for the R environment. *Journal of Statistical Software, 48*(6), 1-29. https://doi.org/10.18637/jss.v048.i06
OpenAlex verification: `https://openalex.org/W1919216659` (DOI `10.18637/jss.v048.i06`).

Thissen, D., Cai, L., & Bock, R. D. (2011). The nominal categories item response model. In M. L. Nering & R. Ostini (Eds.), *Handbook of polytomous item response theory models*. Routledge. https://doi.org/10.4324/9780203861264.ch3
OpenAlex verification: `https://openalex.org/W2344447397` (chapter); parent volume `https://openalex.org/W1540664415` (2011). Publisher and container title confirmed via Crossref. Note a year discrepancy across sources -- OpenAlex dates the chapter record 2015, another record says 2013, and the parent volume is 2011; the volume year is used here. Neither OpenAlex nor Crossref returns a page range, so none is claimed.
Backs the slope-intercept (`internal_parameterization="loading"`) design in `make_2pl_adapter`/`make_gpcm_adapter`: the chapter states that the slope-intercept parameterization is numerically more stable for estimating logistic IRT models than the slope-threshold form, "due to a closer approximation of the likelihood to normality and less error correlation among the parameters". **Scope limit:** that claim is about marginal-maximum-likelihood estimation of IRT models, not about amortized neural posterior estimation. It justifies the parameterization as theoretically better conditioned; it does not establish that the advantage transfers to an NPE flow.

Lord, F. M. (1980). *Applications of item response theory to practical testing problems*. Lawrence Erlbaum.
OpenAlex verification: `https://openalex.org/W1535520578` (Routledge reissue, DOI `10.4324/9780203056615`).
Cited in `scoring.py` for EAP/MAP person scoring.

Thissen, D., & Wainer, H. (Eds.). (2001). *Test scoring*. Lawrence Erlbaum. https://doi.org/10.4324/9781410604729
OpenAlex verification: `https://openalex.org/W4232933324`.
Cited in `scoring.py` for EAP/MAP person scoring.

Talts, S., Betancourt, M., Simpson, D., Vehtari, A., & Gelman, A. (2018). *Validating Bayesian inference algorithms with simulation-based calibration*. arXiv. https://arxiv.org/abs/1804.06788
OpenAlex verification: `https://openalex.org/W2797452937` (DOI `10.48550/arXiv.1804.06788`).
Model card: Theorem 1 -- the rank statistic `sum_l 1[f(theta_l) < f(theta~)]` of an exact posterior is uniform on the integers `[0, L]`, which makes the TARP count `n * f_i` discrete uniform for the same reason. The paper states that SBC "offers no guarantee that the posterior will cover the ground truth for any single observation"; the card's Limitations box rests on that sentence. Backs `sbc.py` calibration and rank diagnostics and the model card.

Schad, D. J., Betancourt, M., & Vasishth, S. (2021). Toward a principled Bayesian workflow in cognitive science. *Psychological Methods, 26*(1), 103–126. https://doi.org/10.1037/met0000275
Crossref verification (OpenAlex search returned no match). Read in the open arXiv preprint (https://arxiv.org/abs/1904.12765v3); Eq. 4 defines the posterior z-score `(posterior mean - true value) / posterior sd`, used by `make_tcc_bias_metric` for the TCC.

Embretson, S. E., & Reise, S. P. (2000). *Item response theory for psychologists*. Lawrence Erlbaum Associates.
No DOI (pre-DOI-era book); verified against the full text.
States that the biserial correlation of an item with total score is directly related to the IRT item discrimination parameter, and that item discrimination increases monotonically with the biserial correlation — the theoretical basis for the `item_rest_correlation` raw feature channel.

DeMars, C. E. (2018). Classical test theory and item response theory. In P. Irwing, T. Booth, & D. J. Hughes (Eds.), *The Wiley handbook of psychometric testing: A multidisciplinary reference on survey, scale and test development* (pp. 49-73). John Wiley & Sons. https://doi.org/10.1002/9781118489772.ch2
OpenAlex verification: `https://openalex.org/W2793300950`.
Defines item discrimination as the point-biserial correlation between item score and total score, and specifies that it is "usually... computed without including x_i in X, termed the corrected item-total correlation" — the source for using item-*rest* (not item-total) correlation to avoid part-whole correlation inflation in `item_rest_correlation`.

Feuerstahler, L. M. (2018). Sources of error in IRT trait estimation. *Applied Psychological Measurement, 42*(5), 359–375. https://doi.org/10.1177/0146621617733955
OpenAlex verification: `https://openalex.org/W2762763459` (OpenAlex lists 2017, the online-first year; the issue is 2018).
Model card: reports bias and CI coverage at fixed generating theta values; EAP with an informative prior shows inward bias and under-coverage at the extremes. Basis for the card's theta-conditional person metrics, shown as descriptive and never as pass/fail.

Falk, C. F., & Feuerstahler, L. M. (2021). On the performance of semi- and nonparametric item response functions in computer adaptive tests. *Educational and Psychological Measurement, 82*(1), 57–75. https://doi.org/10.1177/00131644211014261
OpenAlex verification: `https://openalex.org/W3162676432`.
Model card (p. 9): defines IRF recovery as `RIMSE = 100 * sqrt(sum_q (P_hat(theta_q) - P(theta_q))^2 phi(theta_q) / sum_q phi(theta_q))` with Q = 101 nodes on [-5, 5]; the card's IRF-recovery metric.

Ramsay, J. O. (1991). Kernel smoothing approaches to nonparametric item characteristic curve estimation. *Psychometrika, 56*(4), 611–630. https://doi.org/10.1007/BF02294494
OpenAlex verification: `https://openalex.org/W1979603005`.
Model card (p. 622): defines `RIMSE = [sum_n (P_hat(theta_n) - P(theta_n))^2 phi(theta_n) / sum_n phi(theta_n)]^(1/2)` over N ability values, without a factor; the factor 100 in the card's RIMSE is Falk and Feuerstahler's (2021) scaling. Confirms their attribution.

de la Torre, J., Sijtsma, K., & Kreitchmann, R. S. (2026). Revisiting reliability and measurement precision: Towards a unified perspective. *British Journal of Mathematical and Statistical Psychology*. Advance online publication. https://doi.org/10.1111/bmsp.70068
OpenAlex verification: `https://openalex.org/W7168340650`.
Model card: Eq. 18, the EAP standard error is the posterior SD; Eq. 19, the empirical error variance is its mean square over persons. Source for the card's posterior-based reliability `1 - sigma_E^2 / sigma_theta^2`.

Lee, J.-H. (2026). *Reliability-targeted simulation of item response data: Solving the inverse design problem* (Version 2) [Preprint]. arXiv. https://doi.org/10.48550/arXiv.2512.16012
OpenAlex verification: `https://openalex.org/W4417529464` (OpenAlex lists 2025, the year of version 1; version 2 was read).
Model card (Sec. 2.2, Eq. 2.3–2.4): MSEM-based marginal reliability `w = sigma^2 / (sigma^2 + E[1/J(theta)])` and the average-information variant. Source for the card's information-based reliability of the generating parameters.

## Simulation-based calibration and posterior accuracy diagnostics

Backs the calibration diagnostics in `sbc.py` and the model card: which
calibration quantity to measure and what each one can and cannot detect. Talts
et al. (2018), the SBC foundation, is listed under "IRT foundations and
diagnostics" above.

Säilynoja, T., Bürkner, P.-C., & Vehtari, A. (2022). Graphical test for discrete uniformity and its applications in goodness-of-fit evaluation and multiple sample comparison. *Statistics and Computing, 32*(2), Article 32. https://doi.org/10.1007/s11222-022-10090-6
OpenAlex verification: `https://openalex.org/W3217126915`.
Simultaneous confidence bands for the whole ECDF of PIT values. Cited for the claim that the ECDF band conveys the *nature* of a discrepancy — location bias, over- and under-dispersion leave different signatures — rather than only a pass/fail verdict.
Model card: for empirical PIT values `u_i = (1/S) sum_j I(x_j <= y_i)` from independent draws, `u_i` is discrete uniform on `S + 1` values; the paper gives simultaneous ECDF bands (Sec. 2.2) and recommends the ECDF-*difference* plot for a larger dynamic range. Cited for the TARP ECDF-difference panel: the TARP fractions have this form and are independent across simulations. BayesFlow's `bayesflow.utils.ecdf.simultaneous_ecdf_bands` implements Sec. 2.2 with a continuous-uniform null, off from the discrete null by about `1 / (n + 1)`. A passing band says the curve is compatible with uniformity, nothing more.

Wilson, E. B. (1927). Probable inference, the law of succession, and statistical inference. *Journal of the American Statistical Association, 22*(158), 209–212. https://doi.org/10.1080/01621459.1927.10502953
OpenAlex verification: `https://openalex.org/W2020018978`.

Newcombe, R. G. (1998). Two-sided confidence intervals for the single proportion: Comparison of seven methods. *Statistics in Medicine, 17*(8), 857–872. https://doi.org/10.1002/(SICI)1097-0258(19980430)17:8<857::AID-SIM777>3.0.CO;2-E
OpenAlex verification: `https://openalex.org/W2146272590`.
Back the score interval for the theta-bin coverage in the model card (`card_render.wilson_interval`). Implemented through SciPy's `binomtest(...).proportion_ci(method="wilson")`, whose documentation cites these two papers; the test checks it against the closed form (Newcombe's method 3). Cited only for the interval's formula and that, unlike `p ± z SE`, it stays inside [0, 1] with non-zero width at 0 or n successes.

Modrák, M., Moon, A. H., Kim, S., Bürkner, P., Huurre, N., Faltejsková, K., Gelman, A., & Vehtari, A. (2025). Simulation-based calibration checking for Bayesian computation: The choice of test quantities shapes sensitivity. *Bayesian Analysis, 20*(2), 461–488. https://doi.org/10.1214/23-BA1404
OpenAlex verification: `https://openalex.org/W4388952075` (recorded there with publication year 2023, the online-first date; the journal issue is 2025).
The load-bearing reference for this project's calibration guidance. Sec. 4 and Fig. 4 show that parameter-only test quantities **cannot** detect a posterior equal to the prior, because ranks in different data regions deviate in opposite directions and cancel exactly to uniform. Sec. 6.1 recommends using the individual model parameters together with the joint likelihood of the data as a default. Sec. 6.2 leaves multiple-comparison correction across test quantities as future work. Also supplies the `calibration_log_gamma` statistic used by BayesFlow.

Lemos, P., Coogan, A., Hezaveh, Y., & Perreault-Levasseur, L. (2023). Sampling-based accuracy testing of posterior estimators for general inference. In *Proceedings of the 40th International Conference on Machine Learning* (PMLR 202, pp. 19256–19273). https://proceedings.mlr.press/v202/lemos23a.html
OpenAlex verification: `https://openalex.org/W4319453761` (arXiv DOI record `10.48550/arXiv.2302.03026`; the PMLR version has no separate OpenAlex record).
Introduces TARP. Sec. 3.1 derives the same blind spot as Modrák et al. for coverage rather than ranks: if the estimator returns the prior, the HPD region stops depending on the data and expected coverage is perfect at every level. Thm. 3 gives the remedy — correct expected coverage for *positionable* credible regions is necessary **and** sufficient for the estimator to equal the true posterior. Sample-based, so applicable to flow matching, and joint rather than marginal. Reference implementation: https://github.com/Ciela-Institute/tarp.
Model card: Algorithm 2 defines `f_i = (1/n) sum_j 1[d(theta_ij, theta_r) < d(theta*_i, theta_r)]` and `ECP(alpha) = (1/N) sum_i 1(f_i < 1 - alpha)`, so the ECP curve is the ECDF of the `f_i` at the credibility level and the exact posterior gives the diagonal (Lemos et al. plot ECP against credibility level). Sec. 4.3: with reference points independent of `x`, TARP, like HPD coverage, gives perfect coverage for an estimator that returns the prior; reference points that depend on `x` do not. The card therefore draws both curves. The paper reports no confidence band. `compute_tarp_coverage` implements Algorithm 2.

## Simulation study methodology and reporting

Morris, T. P., White, I. R., & Crowther, M. J. (2019). Using simulation studies to evaluate statistical methods. *Statistics in Medicine, 38*(11), 2074–2102. https://doi.org/10.1002/sim.8086
OpenAlex verification: `https://openalex.org/W2775363893`.
Defines performance measures (bias, empirical SE, MSE, RMSE, coverage) and their Monte Carlo standard errors (Table 6). MCSE formulas used here follow the "generic statistic G" approach: MCSE = SD(G_r)/√n_sim where G_r is the per-replication aggregate. Section 5.3 ("Sample size for simulation studies") gives the corresponding sample-size formula n_sim = Var(θ̂)/MCSE_target² (worked example: Var(θ̂) ≤ 0.04, MCSE_target = 0.005 ⟹ n_sim = 1600)).

Siepe, B. S., Bartoš, F., Morris, T. P., Boulesteix, A.-L., Heck, D. W., & Pawel, S. (2024). Simulation studies for methodological research in psychology: A standardized template for planning, preregistration, and reporting. *Psychological Methods*. Advance online publication. https://doi.org/10.1037/met0000695
OpenAlex verification: `https://openalex.org/W4404372720` (DOI `10.1037/met0000695`). Supersedes the 2023 PsyArXiv preprint `https://openalex.org/W4388133916` (DOI `10.31234/osf.io/ufgy6`).
**Anchor caveat:** the `p. 8` and `Table 3` locators used throughout the codebase (`sbc.py` and `card_metrics.py`) were read against the **preprint**. Table and page numbering may differ in the published article. Cite the 2024 version for the work, and re-verify the locators against it before restating them.
Table 3 (adapted from Morris et al., 2019, Table 6) provides consolidated MCSE formulas including the binomial MCSE for proportions: √(p̂(1−p̂)/n_total). The same table's "Mean of generic statistic G" row gives n_sim = S_G²/MCSE_target², the direct generalization of Morris et al. (2019)'s bias-specific sample-size formula.

Cameron, A. C., & Miller, D. L. (2015). A practitioner's guide to cluster-robust inference. *Journal of Human Resources, 50*(2), 317–372. https://doi.org/10.3368/jhr.50.2.317
OpenAlex verification: `https://openalex.org/W2118836230`. Read from the authors' February 2015 version.
Model card (`card._cluster_mean`, evaluation format 2): the theta-bin MCSEs are the cluster-robust sandwich variance of Sec. II.C, Eqs. 10-11, applied to an intercept-only model with each simulated dataset as one cluster (its persons share the item draws), with the finite-sample factor of Eq. 12, which is `G / (G - 1)` for K = 1. Consistency needs many clusters (`G -> infinity`); Sec. VI covers the few-cluster case.

Bland, M., & Kanaan, M. (2021). Cluster randomized designs. In K. Kim, F. Bretz, Y. K. Cheung, & L. V. Hampson (Eds.), *Handbook of statistical methods for randomized controlled trials* (pp. 191–213). Chapman and Hall/CRC. https://doi.org/10.1201/9781315119694
OpenAlex verification (volume): `https://openalex.org/W3183381669`.
Model card (`card_render.MIN_BIN_CLUSTERS`): Sec. 10.6 states that robust standard errors are a large-sample approximation that breaks down with few clusters, with about 30 clusters as a guideline; the card greys out format-2 theta bins with fewer contributing datasets.

## Person scoring by grid quadrature

Bock, R. D., & Mislevy, R. J. (1982). Adaptive EAP estimation of ability in a microcomputer environment. *Applied Psychological Measurement, 6*(4), 431-444. https://doi.org/10.1177/014662168200600405
OpenAlex verification: `https://openalex.org/W2070351096`.
Backs the expected-a-posteriori (EAP) scoring of theta as a posterior computed on a quadrature grid, as used by `PersonScorer`. Cited for the grid-posterior/EAP idea only; the paper's own quadrature rule is not reproduced.

Naylor, J. C., & Smith, A. F. M. (1982). Applications of a method for the efficient computation of posterior distributions. *Journal of the Royal Statistical Society: Series C (Applied Statistics), 31*(3), 214-225. https://doi.org/10.2307/2347995
OpenAlex verification: `https://openalex.org/W2800420890` (OpenAlex lists only the first page; the page range 214-225 is taken from the article's first page).
Cited for the idea of centring and scaling the quadrature grid on the posterior's own location and scale (adaptive quadrature), which `PersonScorer.score(adaptive_grid=True)` applies with a uniform rule instead of Gauss-Hermite nodes. Checked against the full text: Section 2.2, p. 216, places the Gauss-Hermite nodes at z_i = mu + sqrt(2) sigma t_i with the posterior mean and variance substituted for mu and sigma^2, and refines those moments iteratively, each iteration building the next grid. The coarse-then-adaptive grid in `PersonScorer` follows the same two-step idea; the uniform rule and its span are not from the paper.

Schilling, S., & Bock, R. D. (2005). High-dimensional maximum marginal likelihood item factor analysis by adaptive quadrature. *Psychometrika, 70*(3), 533-555. https://doi.org/10.1007/s11336-003-1141-x
OpenAlex verification: `https://openalex.org/W2136244180`.
Cited for adaptive quadrature in an IRT setting (quadrature nodes centred on each respondent's posterior). Checked against the full text: Section 2.3 applies Naylor and Smith's (1982) adaptive quadrature to each response pattern's posterior density of theta, and the discussion reports improved accuracy from adapting the quadrature points to the individual posterior distributions rather than using a fixed grid. It does not verify the specific span (default 4 coarse SDs), point count (default 21) or the half-coarse-spacing SD floor used in `PersonScorer`; those are implementation choices, checked by the tests in `tests/test_scoring.py` (`TestAdaptiveGrid`).

Devroye, L. (1986). *Non-uniform random variate generation*. Springer. https://doi.org/10.1007/978-1-4613-8643-8
OpenAlex verification: `https://openalex.org/W2152828142` (checked by DOI).
Checked against the full text: Chapter II, Section 2.1, Theorem 2.1, p. 28: for a continuous distribution function F, F^-1(U) with U uniform on [0, 1] has distribution F. The CDF of a piecewise-constant density is continuous, so the theorem applies; the book does not single out piecewise-constant densities for this (its "histogram method", p. 103, is a discrete-distribution method).
Cited for the inversion method (inverse CDF) used to draw plausible values and quantiles from a piecewise-constant density: with each grid mass uniform on its cell the CDF is piecewise linear and is inverted by linear interpolation. The statement that this removes the half-cell low bias of "CDF at the grid point" quantiles is a direct consequence of that construction and is checked numerically in `TestCellUniformQuantiles` and the calibration test, not taken from the source.

## Simulation algorithms

Maddison, C. J., Tarlow, D., & Minka, T. (2014). *A\* sampling*. In *Advances in Neural Information Processing Systems* (Vol. 27). https://www.microsoft.com/en-us/research/wp-content/uploads/2016/02/5449-a-sampling.pdf
OpenAlex verification: `https://openalex.org/W2296701362` (no DOI registered; NeurIPS 27 proceedings).
Used for the Gumbel-max trick and its exponential-noise representation for exact categorical sampling.

## Memory controls

Chen, T., Xu, B., Zhang, C., & Guestrin, C. (2016). *Training deep nets with sublinear memory cost*. arXiv. https://arxiv.org/abs/1604.06174
OpenAlex verification: `https://openalex.org/W2338908902` (DOI `10.48550/arXiv.1604.06174`).

## Package documentation used for implementation details

Okabe, M., & Ito, K. (2008). *Color universal design (CUD): How to make figures and presentations that are friendly to colorblind people*. https://jfly.uni-koeln.de/color/ — the categorical palette of the model-card dashboard's point plots; hex values as in R's `grDevices::palette.colors("Okabe-Ito")`.

Keras Team. (2026). *Model config serialization* (`get_config` / `from_config`). https://keras.io/api/models/model_saving_apis/model_config_serialization/

Keras Team. (2026). *LayerNormalization layer*. https://keras.io/api/layers/normalization_layers/layer_normalization/

Keras Team. (2026). *Making new layers and models via subclassing* (`add_weight`, `build`, `call`, `get_config`). https://keras.io/guides/making_new_layers_and_models_via_subclassing/

TensorFlow Team. (2026). *Save, serialize, and export models* (custom object serialization behavior). https://www.tensorflow.org/guide/keras/serialization_and_saving

Keras Team. (2026). *Mixed precision*. https://keras.io/api/mixed_precision/

PyTorch Team. (2026). *torch.utils.checkpoint*. https://docs.pytorch.org/docs/stable/checkpoint.html

PyTorch Team. (2026). *torch.set_float32_matmul_precision*. https://docs.pytorch.org/docs/stable/generated/torch.set_float32_matmul_precision.html

PyTorch Team. (2026). *CUDA semantics* (memory allocator options, including `PYTORCH_CUDA_ALLOC_CONF`). https://docs.pytorch.org/docs/stable/notes/cuda.html#optimizing-memory-usage-with-pytorch-cuda-alloc-conf

PyTorch Team. (2026). *torch.cuda.memory.set_per_process_memory_fraction* (caps the caching allocator to a fraction of total device memory; raises `torch.cuda.OutOfMemoryError` in the allocator rather than deferring to the platform driver once the cap would be exceeded). https://docs.pytorch.org/docs/stable/generated/torch.cuda.memory.set_per_process_memory_fraction.html

NumPy Developers. (2026). *Parallel random number generation: sequence of integer seeds*. https://numpy.org/doc/stable/reference/random/parallel.html#sequence-of-integer-seeds
    Backs `compute_tarp_coverage`'s reference-seed salting: a
    `[varying, fixed]` integer sequence carries "about the same"
    collision safety as spawning, because `SeedSequence` uses integer
    hashes with avalanche properties so close input seeds produce
    initial states "very far from each other (with very high
    probability)".

Tate, R. F. (1955). The theory of the correlation between two continuous variables when one is dichotomized. *Biometrika, 42*(1–2), 205–216. https://doi.org/10.1093/biomet/42.1-2.205
OpenAlex verification: `https://openalex.org/W1983724935`.
The primary theoretical source for the biserial correlation — the estimate of what the Pearson correlation would be had the dichotomized variable been observed continuously. Backs the correction `r_bis = r_pb * sqrt(p(1-p)) / phi(z_p)`, with `phi` the standard normal density at `z_p = Phi^-1(1-p)`. The same relation is stated in NCSS's and Winsteps' documentation, neither of which cites a source; this is the citation those describe. **This package deliberately does not apply the correction** — `classical_item_statistics` returns the attenuated `point_biserial`, because a reference point that estimates the parameter well degenerates as a TARP reference (see that function's docstring). The choice costs real information and is made on purpose: in a synthetic check the corrected index tracks the normal-ogive `rho` closely and stably across priors, while the attenuated one does not and is strongly prior-dependent — its correlation moves across most of the unit interval with the spread of `b` alone, because difficulty drives the proportion correct and the proportion correct is exactly what the correction divides out. No specific pair of figures is quoted here; only the direction is claimed, and the direction is what the decision rests on.

Martinková, P., & Hladká, A. (2023). *Computational aspects of psychometric methods: With R*. Chapman and Hall/CRC. https://doi.org/10.1201/9781003054313
OpenAlex verification: `https://openalex.org/W4376647594`.
Sec. 7.3.1 reports Lord and Novick's (1968, Chapter 16) normal-ogive correspondence used to justify the TARP reference coordinates: under a standard normal latent trait, `rho_j = a_j / sqrt(1 + a_j^2)`, hence `a_j = rho_j / sqrt(1 - rho_j^2)`, and `b_j = Phi^-1(pi_j) / rho_j`. Two cautions, from synthetic checks rather than inherited: the correspondence is derived for the **normal-ogive** model while this package uses a **logistic** link, so it holds only up to the metric scaling (`z_difficulty` against `b_j * rho_j` is monotone but not an identity); and `pi_j` must be read as the proportion **incorrect** under `P(correct) = logistic(a_j (theta - b_j))`, since `Phi^-1(1 - p) / rho` correlates positively with `b` while `Phi^-1(p) / rho` correlates negatively.

Bowling, S. R., Khasawneh, M. T., Kaewkuekool, S., & Cho, B. R. (2009). A logistic approximation to the cumulative normal distribution. *Journal of Industrial Engineering and Management, 2*(1), 114–127. https://doi.org/10.3926/jiem.2009.v2n1.p114-127
OpenAlex verification: `https://openalex.org/W4251385712`.
Maps the logistic and normal scales with a **function** rather than a single constant: `Phi(x) ~= inv_logit(0.07056 x^3 + 1.5976 x)`. This is what Stan's `Phi_approx` implements, and Stan's functions reference cites this paper for it. Verified numerically here over `x` in [-6, 6]: max absolute error **0.000141**, matching Stan's documented 0.00014, against **0.0095** for the traditional `inv_logit(1.702 x)` — about 67x better.

Two cautions recorded, because the constants are easy to misapply. Savalei's (2006) `D = 1.749` minimises Kullback–Leibler divergence, not maximum error, and on max absolute error it is **worse** than 1.702 (0.0150 vs 0.0095); the two answer different questions and neither dominates. And Camilli (2017) shows by simulation that no single `D` is correct: the ratio of logistic to normal-ogive discrimination ranged 1.666 to 2.031 across items, varying systematically with `a` and much more strongly for tail values of `b`, so `D` is "an average scaling value rather than a scaling constant". That matters for this package specifically, whose default `b ~ Normal(0, 2)` generates exactly the extreme difficulties where the approximation degrades — so a constant rescaling should not be applied to `classical_item_statistics` output without checking the per-item structure first.

Camilli, G. (2017). The scaling constant D in item response theory. *Open Journal of Statistics, 7*(5), 780–785. https://doi.org/10.4236/ojs.2017.75055
OpenAlex verification: `https://openalex.org/W2761290186`. See the entry for Bowling et al. (2009) for what this contributes. Cited here for the finding that the logistic-to-normal-ogive scaling varies across items rather than being constant, which bounds how far the correspondence in Martinkova and Hladka (2023) can be pushed.

## Amortized inference software and multilevel designs

Habermann, D., Bulling, A., Radev, S. T., & Bürkner, P.-C. (2026). *Amortized Bayesian inference on multilevel models of arbitrary structure*. arXiv. https://doi.org/10.48550/arXiv.2609.40024
OpenAlex verification: `https://openalex.org/W7215088702`.
Backs `src/bayesflow_irt/draws_summary.py` and the autoregressive item posterior of `EquivariantIRTApproximator` (`item_posterior="autoregressive_coupling_flow"`): autoregressive inference for crossed grouping factors such as items x persons (Sec. 2.5, Eq. 3) and per-network summaries with grouping-wise means (Sec. 2.6); the draws summary uses extended observations and log counts as conditions.

Kühmichel, L., Huang, J. M., Pratz, V., Arruda, J., Olischläger, H., Habermann, D., Kucharsky, S., Elsemüller, L., Mishra, A., Bracher, N., Jedhoff, S., Schmitt, M., Bürkner, P.-C., & Radev, S. T. (2026). *BayesFlow 2: Multi-backend amortized Bayesian inference in Python*. arXiv. https://doi.org/10.48550/arXiv.2602.07098
OpenAlex verification: `https://openalex.org/W7128506243`.
Sec. 4.5 documents `fit_online(epochs, num_batches_per_epoch, batch_size)` and states that online training specifies the number of batches per epoch as the number of training steps, "instead of computing the training steps using the simulation budget". Cited on the model-card dashboard as the source for neural posterior estimation with BayesFlow.
