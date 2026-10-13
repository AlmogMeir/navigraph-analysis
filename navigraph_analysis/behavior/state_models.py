"""Model fitting and cross-family comparison for behavioral-state HMMs.

Companion to :mod:`navigraph_analysis.behavior.states`, which builds the
per-node-visit feature frame, the weak-label anchors and
:func:`~navigraph_analysis.behavior.states.validate_states` and stays
dependency-light. This module holds everything that needs the heavier optional
packages: fitting several HMM families on the *same* feature frame and scoring
them on a shared set of metrics, so they are directly comparable rather than each
notebook reporting its own family's numbers.

Ported from ``navigraph.analysis.state_models`` (developed in ``state_hmm.ipynb``
and ``state_model_comparison.ipynb``).

Families
--------
- ``IO-HMM`` (:func:`iohmm_scores`) — input-driven (``IOHMM`` package):
  covariates -> state -> ``turn_type`` emission. No persistence prior.
- ``Sticky-Categorical`` (:func:`categorical_scores`) — ``hmmlearn``
  ``CategoricalHMM`` on the ``turn_type`` sequence alone (no covariates), with a
  diagonal-heavy ``transmat_prior`` for stickiness.
- ``Sticky-Gaussian`` / plain ``Gaussian`` (:func:`gaussian_scores`) —
  ``hmmlearn`` ``GaussianHMM`` on the continuous covariate matrix, with an
  optional sticky ``transmat_prior``.
- ``ssm GLM-HMM (sticky)`` (:func:`glmhmm_scores`) — the ``ssm`` package's true
  input-driven GLM-HMM with a built-in sticky transition prior. Optional: guarded
  by :data:`HAS_SSM`, since ``ssm`` needs a C++ compiler to install.

All four ``*_scores`` functions return the same dict schema —
``{family, K, ll, aic, bic, ll_test, seed, model, states}`` — so downstream
comparison code never has to branch per family.

Stickiness
----------
:func:`sticky_transmat_prior` and ``ssm``'s ``StickyTransitions`` use the
mathematically equivalent pseudo-count mechanism (confirmed against both
libraries' source): ``kappa`` extra "stayed in this state" counts are added to the
diagonal of the transition-count matrix before normalizing. Using the same
``kappa`` (default 100, ``ssm``'s own default) in both keeps the hmmlearn-based
and ssm-based sticky variants on equal footing.

Dependencies
------------
``pip install 'navigraph-analysis[neural]'`` for the hmmlearn/IOHMM families.
``ssm`` is not on PyPI under that name and needs a compiler::

    sudo dnf install -y gcc-c++
    pip install "git+https://github.com/lindermanlab/ssm.git"

Every fitter raises a clear message naming what to install, so importing this
module works with none of them present.
"""

from __future__ import annotations

import signal
from contextlib import contextmanager
from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

from .turns import TURN_CATEGORIES

#: ``ssm``'s own StickyTransitions default; reused for the hmmlearn families.
STICKY_KAPPA = 100
#: Hard wall-clock cap per model fit (see :func:`time_limit`).
DEFAULT_TIMEOUT_S = 90


# ---------------------------------------------------------------------------
# Optional heavy deps -- probed lazily so importing this module always works
# ---------------------------------------------------------------------------

def _require_hmmlearn():
    try:
        from hmmlearn.hmm import GaussianHMM, CategoricalHMM
    except ImportError as exc:  # pragma: no cover - depends on optional install
        raise ImportError(
            "hmmlearn is required for the Gaussian / Categorical HMM families. "
            "Install with: pip install 'navigraph-analysis[neural]'"
        ) from exc
    return GaussianHMM, CategoricalHMM


def _require_sklearn():
    try:
        from sklearn.linear_model import LogisticRegression
        from sklearn.metrics import accuracy_score
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "scikit-learn is required for turn_prediction_uplift. "
            "Install with: pip install 'navigraph-analysis[neural]'"
        ) from exc
    return LogisticRegression, accuracy_score


def apply_iohmm_sklearn_shim() -> None:
    """Make IOHMM 0.0.7 work with scikit-learn >= 1.3.

    IOHMM calls sklearn's ``label_binarize`` positionally; sklearn >= 1.3 made
    ``classes`` keyword-only. Patch once; idempotent.
    """
    import sklearn.preprocessing as _skp
    import IOHMM.linear_models as _iolm

    orig_lb = _skp.label_binarize
    _iolm.label_binarize = lambda Y, classes, **k: orig_lb(Y, classes=classes, **k)


def _require_iohmm():
    try:
        apply_iohmm_sklearn_shim()
        from IOHMM import UnSupervisedIOHMM, DiscreteMNL, CrossEntropyMNL
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "IOHMM is required for the IO-HMM family. "
            "Install with: pip install 'navigraph-analysis[neural]'"
        ) from exc
    return UnSupervisedIOHMM, DiscreteMNL, CrossEntropyMNL


def _have(fn) -> bool:
    try:
        fn()
        return True
    except ImportError:
        return False


def _ssm_module():
    import ssm
    return ssm


#: True when the ``ssm`` package is importable (needed for :func:`glmhmm_scores`).
try:  # pragma: no cover - depends on optional install
    import ssm as _ssm
    HAS_SSM = True
except Exception:
    _ssm = None
    HAS_SSM = False

#: True when hmmlearn is importable.
HAS_HMMLEARN = _have(_require_hmmlearn)
#: True when IOHMM (and its sklearn shim) is importable.
HAS_IOHMM = _have(_require_iohmm)


_SSM_INSTALL_HINT = (
    "ssm is not installed. Install with:\n"
    "  sudo dnf install -y gcc-c++\n"
    '  pip install "git+https://github.com/lindermanlab/ssm.git"\n'
    "(PyPI's `ssm` package is a different, unrelated, non-building package -- do not use it.)"
)


# ---------------------------------------------------------------------------
# Infra: wall-clock guard, held-out split, turn encoding, sticky prior
# ---------------------------------------------------------------------------

@contextmanager
def time_limit(seconds: float):
    """Hard wall-clock cap for a single model fit (Unix only, SIGALRM-based).

    A handful of ``(K, seed)`` combinations can pathologically slow an EM fit;
    this bounds a K-sweep's worst case regardless of cause. On a platform without
    ``SIGALRM`` the cap is skipped rather than failing the fit.
    """
    if not hasattr(signal, "SIGALRM"):  # pragma: no cover - non-Unix
        yield
        return

    def _handler(signum, frame):
        raise TimeoutError(f"fit exceeded {seconds}s")

    old = signal.signal(signal.SIGALRM, _handler)
    signal.alarm(int(seconds))
    try:
        yield
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, old)


def trial_blocked_test_mask(trial_idx: pd.Series, every: int = 5) -> np.ndarray:
    """Boolean test mask holding out every *every*-th trial id.

    Blocked by trial rather than by node-visit: consecutive node-visits inside a
    trial are strongly dependent, so a random split would leak.
    """
    uniq = np.sort(pd.unique(trial_idx))
    test_trials = set(uniq[::every])
    return np.isin(trial_idx, list(test_trials))


def encode_turns(turn_type: pd.Series) -> np.ndarray:
    """Integer-encode ``turn_type`` in :data:`TURN_CATEGORIES` order (0..4)."""
    cat = pd.Categorical(turn_type, categories=TURN_CATEGORIES)
    return cat.codes.astype(int)


def sticky_transmat_prior(K: int, kappa: float = STICKY_KAPPA) -> np.ndarray:
    """Diagonal-heavy Dirichlet prior over transition-matrix rows.

    ``hmmlearn``'s M-step is ``transmat_ = max(transmat_prior - 1 + counts, 0)``
    (see ``hmmlearn/base.py``), so this adds exactly *kappa* pseudo "stayed in this
    state" counts to the diagonal — the same mechanism as ``ssm``'s
    ``StickyTransitions`` (``expected_joints += kappa * eye(K)``).
    """
    return np.eye(K) * kappa + 1.0


def _best_over_seeds(fit, score, seeds, family_label: str, K: int):
    """Fit once per seed and keep the highest-likelihood model.

    Seeds that raise or time out are skipped; if every seed fails that is a real
    result about this ``(family, K)`` and is raised rather than returned as NaN.
    """
    best = None
    for s in seeds:
        try:
            m = fit(s)
            ll = score(m)
        except Exception:
            continue
        if best is None or ll > best[0]:
            best = (ll, m, s)
    if best is None:
        raise RuntimeError(
            f"{family_label}: all {len(list(seeds))} seeds failed/timed out at K={K}"
        )
    return best


# ---------------------------------------------------------------------------
# Family: IO-HMM (input-driven, IOHMM package) -- reference, no sticky prior
# ---------------------------------------------------------------------------

def _io_frame(Xmat: np.ndarray, turn: np.ndarray, covs: Sequence[str]) -> pd.DataFrame:
    d = pd.DataFrame(Xmat, columns=list(covs))
    d["turn"] = turn
    return d


def fit_iohmm(K, Xmat, turn, covs, seed=0, max_em_iter=25, em_tol=1e-3,
              inner_max_iter=50, alpha=1.0, timeout_s=DEFAULT_TIMEOUT_S):
    """Fit one unsupervised IO-HMM: covariates -> state -> ``turn_type``.

    *alpha* is the L2 regularization strength for ``DiscreteMNL`` /
    ``CrossEntropyMNL``. IOHMM defaults to ``alpha=0`` (no regularization —
    internally ``C = 1/alpha -> inf``), which lets coefficients blow up toward
    perfect separation at ``K >= 4`` and crashes every seed. ``alpha=1.0`` keeps
    them bounded (verified: 15/15 seeds converge at K=3/4/5).
    """
    UnSupervisedIOHMM, DiscreteMNL, CrossEntropyMNL = _require_iohmm()
    np.random.seed(seed)
    mod = UnSupervisedIOHMM(num_states=K, max_EM_iter=max_em_iter, EM_tol=em_tol)
    mod.set_models(
        model_emissions=[DiscreteMNL(reg_method="l2", alpha=alpha, max_iter=inner_max_iter)],
        model_transition=CrossEntropyMNL(solver="lbfgs", reg_method="l2", alpha=alpha,
                                         max_iter=inner_max_iter),
        model_initial=CrossEntropyMNL(solver="lbfgs", reg_method="l2", alpha=alpha,
                                      max_iter=inner_max_iter),
    )
    mod.set_inputs(covariates_initial=[], covariates_transition=list(covs),
                   covariates_emissions=[list(covs)])
    mod.set_outputs([["turn"]])
    mod.set_data([_io_frame(Xmat, turn, covs)])
    with time_limit(timeout_s):
        mod.train()
    return mod


def io_states(mod) -> np.ndarray:
    """Most likely state per node-visit from a fitted IO-HMM."""
    return np.array(mod.log_gammas[0]).argmax(1)


def iohmm_scores(K, X, turn_out, covs, seeds=range(5), family_label="IO-HMM",
                 timeout_s=DEFAULT_TIMEOUT_S, alpha=1.0) -> Dict:
    """Best-of-seeds IO-HMM at *K* states, scored on the shared schema."""
    ll, mod, s = _best_over_seeds(
        lambda seed: fit_iohmm(K, X, turn_out, covs, seed=seed,
                               timeout_s=timeout_s, alpha=alpha),
        lambda m: float(np.array(m.log_likelihoods)[-1]),
        seeds, family_label, K,
    )
    states = io_states(mod)
    p = X.shape[1]
    C = len(set(turn_out))
    n_params = K * C * (p + 1) + K * K * (p + 1) + (K - 1)
    return dict(family=family_label, K=K, ll=ll,
                aic=-2 * ll + 2 * n_params,
                bic=-2 * ll + n_params * np.log(len(X)),
                ll_test=np.nan, seed=s, model=mod, states=states)


# ---------------------------------------------------------------------------
# Family: Gaussian-HMM (hmmlearn), plain or sticky
# ---------------------------------------------------------------------------

def fit_gaussian(K, Xmat, seed=0, n_iter=150, timeout_s=DEFAULT_TIMEOUT_S,
                 transmat_prior: Optional[np.ndarray] = None):
    """Fit a diagonal-covariance ``GaussianHMM`` on the covariate matrix."""
    GaussianHMM, _ = _require_hmmlearn()
    kwargs = dict(n_components=K, covariance_type="diag", n_iter=n_iter, random_state=seed)
    if transmat_prior is not None:
        kwargs["transmat_prior"] = transmat_prior
    m = GaussianHMM(**kwargs)
    with time_limit(timeout_s):
        m.fit(Xmat)
    return m


def gaussian_scores(K, X, seeds=range(5), train_mask=None, test_mask=None,
                    transmat_prior: Optional[np.ndarray] = None, family_label="Gaussian",
                    timeout_s=DEFAULT_TIMEOUT_S) -> Dict:
    """Best-of-seeds Gaussian HMM at *K* states, scored on the shared schema.

    Pass both *train_mask* and *test_mask* (see :func:`trial_blocked_test_mask`)
    to also get held-out log-likelihood in ``ll_test``.
    """
    ll, m, s = _best_over_seeds(
        lambda seed: fit_gaussian(K, X, seed=seed, transmat_prior=transmat_prior,
                                  timeout_s=timeout_s),
        lambda mm: mm.score(X),
        seeds, family_label, K,
    )
    states = m.predict(X)
    n_params = K * X.shape[1] * 2 + K * (K - 1) + (K - 1)   # means+vars + trans + init
    ll_test = np.nan
    if train_mask is not None and test_mask is not None:
        try:
            m_tr = fit_gaussian(K, X[train_mask], seed=s, transmat_prior=transmat_prior,
                                timeout_s=timeout_s)
            ll_test = m_tr.score(X[test_mask])
        except Exception:
            pass
    return dict(family=family_label, K=K, ll=ll,
                aic=-2 * ll + 2 * n_params,
                bic=-2 * ll + n_params * np.log(len(X)),
                ll_test=ll_test, seed=s, model=m, states=states)


# ---------------------------------------------------------------------------
# Family: Categorical-HMM on turn_type alone (hmmlearn), plain or sticky
# ---------------------------------------------------------------------------

def fit_categorical(K, turn_codes, seed=0, n_iter=150, timeout_s=DEFAULT_TIMEOUT_S,
                    transmat_prior: Optional[np.ndarray] = None,
                    n_features: Optional[int] = None):
    """Fit a ``CategoricalHMM`` on the integer-encoded ``turn_type`` sequence."""
    _, CategoricalHMM = _require_hmmlearn()
    if n_features is None:
        n_features = len(TURN_CATEGORIES)
    kwargs = dict(n_components=K, n_features=n_features, random_state=seed, n_iter=n_iter)
    if transmat_prior is not None:
        kwargs["transmat_prior"] = transmat_prior
    m = CategoricalHMM(**kwargs)
    with time_limit(timeout_s):
        m.fit(np.asarray(turn_codes).reshape(-1, 1))
    return m


def categorical_scores(K, turn_codes, seeds=range(5), train_mask=None, test_mask=None,
                       transmat_prior: Optional[np.ndarray] = None,
                       n_features: Optional[int] = None,
                       family_label="Sticky-Categorical",
                       timeout_s=DEFAULT_TIMEOUT_S) -> Dict:
    """Best-of-seeds Categorical HMM at *K* states, scored on the shared schema."""
    if n_features is None:
        n_features = len(TURN_CATEGORIES)
    turn_codes = np.asarray(turn_codes)
    Y = turn_codes.reshape(-1, 1)

    ll, m, s = _best_over_seeds(
        lambda seed: fit_categorical(K, turn_codes, seed=seed,
                                     transmat_prior=transmat_prior,
                                     n_features=n_features, timeout_s=timeout_s),
        lambda mm: mm.score(Y),
        seeds, family_label, K,
    )
    states = m.predict(Y)
    n_params = K * (n_features - 1) + K * (K - 1) + (K - 1)   # emission + trans + init
    ll_test = np.nan
    if train_mask is not None and test_mask is not None:
        try:
            m_tr = fit_categorical(K, turn_codes[train_mask], seed=s,
                                   transmat_prior=transmat_prior,
                                   n_features=n_features, timeout_s=timeout_s)
            ll_test = m_tr.score(turn_codes[test_mask].reshape(-1, 1))
        except Exception:
            pass
    return dict(family=family_label, K=K, ll=ll,
                aic=-2 * ll + 2 * n_params,
                bic=-2 * ll + n_params * np.log(len(turn_codes)),
                ll_test=ll_test, seed=s, model=m, states=states)


# ---------------------------------------------------------------------------
# Family: ssm GLM-HMM with sticky transitions (optional, guarded by HAS_SSM)
# ---------------------------------------------------------------------------

def fit_glmhmm(K, X, turn_codes, seed=0, kappa=STICKY_KAPPA, num_iters=150):
    """Fit ``ssm``'s input-driven GLM-HMM with a built-in sticky transition prior."""
    if not HAS_SSM:
        raise RuntimeError(_SSM_INSTALL_HINT)
    np.random.seed(seed)
    turn_codes = np.asarray(turn_codes).reshape(-1, 1)
    C = len(TURN_CATEGORIES)
    model = _ssm.HMM(K, D=1, M=X.shape[1], observations="input_driven_obs",
                     observation_kwargs=dict(C=C),
                     transitions="sticky", transition_kwargs=dict(kappa=kappa))
    model.fit([turn_codes], inputs=[X], method="em", num_iters=num_iters, verbose=0)
    return model


def glmhmm_scores(K, X, turn_codes, seeds=range(5), kappa=STICKY_KAPPA, num_iters=150,
                  family_label="ssm GLM-HMM (sticky)") -> Dict:
    """Best-of-seeds ssm GLM-HMM at *K* states, scored on the shared schema."""
    if not HAS_SSM:
        raise RuntimeError(_SSM_INSTALL_HINT)
    turn_codes = np.asarray(turn_codes)
    turn_col = turn_codes.reshape(-1, 1)

    ll, mod, s = _best_over_seeds(
        lambda seed: fit_glmhmm(K, X, turn_codes, seed=seed, kappa=kappa, num_iters=num_iters),
        lambda m: float(m.log_likelihood([turn_col], inputs=[X])),
        seeds, family_label, K,
    )
    states = np.asarray(mod.most_likely_states(turn_col, input=X))
    C = len(TURN_CATEGORIES)
    p = X.shape[1]
    n_params = K * C * (p + 1) + K * K * (p + 1) + (K - 1)   # parity with IO-HMM's count
    return dict(family=family_label, K=K, ll=ll,
                aic=-2 * ll + 2 * n_params,
                bic=-2 * ll + n_params * np.log(len(turn_codes)),
                ll_test=np.nan, seed=s, model=mod, states=states)


# ---------------------------------------------------------------------------
# Cross-family evaluation (state-column-agnostic)
# ---------------------------------------------------------------------------

def empirical_transition_matrix(states, K: int) -> np.ndarray:
    """Row-normalized count of observed ``state[t] -> state[t+1]`` transitions."""
    states = np.asarray(states)
    mat = np.zeros((K, K))
    for a, b in zip(states[:-1], states[1:]):
        mat[a, b] += 1
    row_sums = mat.sum(axis=1, keepdims=True)
    return np.divide(mat, row_sums, out=np.zeros_like(mat), where=row_sums > 0)


def state_run_lengths(states) -> Dict[int, List[int]]:
    """Per state, the list of consecutive-node-visit run lengths.

    A run is a maximal stretch of consecutive node-visits assigned the same state
    — how many nodes in a row the model says the animal stayed in that state
    before switching. Same run detection as the stickiness block of
    :func:`navigraph_analysis.behavior.states.validate_states`, but returns the
    full distribution rather than only its mean.
    """
    states = np.asarray(states)
    out: Dict[int, List[int]] = {}
    if len(states) == 0:
        return out
    cur_state = states[0]
    run = 1
    for v in states[1:]:
        if v == cur_state:
            run += 1
        else:
            out.setdefault(int(cur_state), []).append(run)
            cur_state = v
            run = 1
    out.setdefault(int(cur_state), []).append(run)
    return out


def alternation_purity(feat: pd.DataFrame, state_col: str):
    """Per-alternation-segment fraction of node-visits in that segment's modal state.

    One value per direct within-patch alternation path. Returns
    ``(purities, dominant_states, seg_ids)``, each of length ``n_segments``.
    """
    seg_ids = feat.loc[feat["anchor_exploit"], "_seg"].unique()
    purities, dominant_states = [], []
    for seg in seg_ids:
        vals = feat.loc[feat["_seg"] == seg, state_col]
        vc = vals.value_counts(normalize=True)
        purities.append(vc.iloc[0])
        dominant_states.append(vc.index[0])
    return np.array(purities), np.array(dominant_states), seg_ids


def change_of_mind_window(feat: pd.DataFrame, state_col: str, window: int = 6):
    """State trace around each change-of-mind apex.

    Columns are relative node-visit offsets ``-window .. +window`` (0 = the apex).
    Returns ``(matrix, apex_idx)``.
    """
    apex_idx = feat.loc[feat["anchor_change_of_mind"], "node_visit_idx"].values
    state_arr = feat.set_index("node_visit_idx")[state_col]
    mat = np.full((len(apex_idx), 2 * window + 1), np.nan)
    for i, a in enumerate(apex_idx):
        for j, off in enumerate(range(-window, window + 1)):
            mat[i, j] = state_arr.get(a + off, np.nan)
    return mat, apex_idx


def turn_prediction_uplift(X, turn_out, state_seq, train_mask, test_mask) -> Dict:
    """Held-out ``turn_type`` accuracy: covariates alone vs covariates + one-hot state.

    Well-defined and comparable across *every* family — not only the ones whose
    emission natively models ``turn_type`` — because it is a downstream logistic
    regression rather than the HMM's own likelihood.
    """
    LogisticRegression, accuracy_score = _require_sklearn()
    y = np.asarray(turn_out)

    lr = LogisticRegression(max_iter=500)
    lr.fit(X[train_mask], y[train_mask])
    acc_cov = accuracy_score(y[test_mask], lr.predict(X[test_mask]))

    onehot = pd.get_dummies(np.asarray(state_seq)).values.astype(float)
    Xs = np.hstack([X, onehot])
    lr2 = LogisticRegression(max_iter=500)
    lr2.fit(Xs[train_mask], y[train_mask])
    acc_state = accuracy_score(y[test_mask], lr2.predict(Xs[test_mask]))

    return dict(acc_covariates_only=acc_cov,
                acc_covariates_plus_state=acc_state,
                uplift=acc_state - acc_cov)


def compare_families(results: Sequence[Dict]) -> pd.DataFrame:
    """Tabulate ``*_scores`` dicts for comparison, best BIC first.

    Drops the fitted ``model`` / ``states`` objects so the table stays printable.
    """
    rows = [{k: v for k, v in r.items() if k not in ("model", "states")} for r in results]
    out = pd.DataFrame(rows)
    return out.sort_values("bic").reset_index(drop=True) if len(out) else out


__all__ = [
    "HAS_SSM", "HAS_HMMLEARN", "HAS_IOHMM",
    "STICKY_KAPPA", "DEFAULT_TIMEOUT_S",
    "apply_iohmm_sklearn_shim", "time_limit", "trial_blocked_test_mask",
    "encode_turns", "sticky_transmat_prior",
    "fit_iohmm", "io_states", "iohmm_scores",
    "fit_gaussian", "gaussian_scores",
    "fit_categorical", "categorical_scores",
    "fit_glmhmm", "glmhmm_scores",
    "empirical_transition_matrix", "state_run_lengths",
    "alternation_purity", "change_of_mind_window",
    "turn_prediction_uplift", "compare_families",
]
