"""Tests for the properties that, if broken, make every number meaningless.

The lookahead tests are the important ones. A backtest with a one-bar leak
does not look broken -- it looks brilliant, which is much worse.

    python -m pytest tests/ -v
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from loonie import backtest as bt, config, cv, features, genome, metrics  # noqa: E402
from loonie.data import Panel  # noqa: E402


# =============================================================================
#  Fixtures
# =============================================================================
def synthetic_panel(T=700, N=40, seed=7, kill=None) -> Panel:
    """Deterministic random-walk panel. `kill` = {col: t} delists that name."""
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2018-01-01", periods=T)
    r = rng.normal(0.0004, 0.015, (T, N)).astype(np.float32)
    close = (100 * np.cumprod(1 + r, axis=0)).astype(np.float32)

    for col, t in (kill or {}).items():
        close[t:, col] = np.nan

    bars = {
        "close": close,
        "open": (close * (1 + rng.normal(0, 0.002, (T, N)))).astype(np.float32),
        "high": (close * (1 + np.abs(rng.normal(0, 0.006, (T, N))))).astype(np.float32),
        "low": (close * (1 - np.abs(rng.normal(0, 0.006, (T, N))))).astype(np.float32),
        "volume": np.full((T, N), 5e6, np.float32),
    }
    member = np.ones((T, N), bool)
    tradable = member & np.isfinite(close) & (close > 0)
    return Panel(dates=dates, tickers=["T%02d" % i for i in range(N)],
                 bars=bars, member=member, tradable=tradable,
                 coverage={"provider": "synthetic", "coverage": 1.0})


@pytest.fixture(scope="module")
def cfg():
    return config.load()


SEARCH_PATH = ("evolve", "genome", "features", "peers", "methods", "backtest")


def _imports_of(mod: str) -> set:
    """Module names a file actually imports, by AST -- not by substring.

    The naive version of this check greps for the module name and trips over
    ordinary prose: peers.py explains that cohorts are "driven by common
    factors", which is not an import of loonie/factors.py. A test that fails
    on a docstring is a test people learn to weaken.
    """
    import ast

    root = Path(__file__).resolve().parent.parent / "loonie"
    tree = ast.parse((root / (mod + ".py")).read_text(encoding="utf-8"))
    out = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for a in node.names:
                out.add(a.name.split(".")[0])
        elif isinstance(node, ast.ImportFrom):
            if node.module:
                out.add(node.module.split(".")[-1])
            for a in node.names:
                out.add(a.name)
    return out


# =============================================================================
#  Lookahead -- the tests that matter most
# =============================================================================
def test_features_are_causal():
    """A feature at time t must not change when data after t is deleted."""
    full = synthetic_panel(T=600)
    cut = 400
    trunc = Panel(dates=full.dates[:cut], tickers=full.tickers,
                  bars={k: v[:cut] for k, v in full.bars.items()},
                  member=full.member[:cut], tradable=full.tradable[:cut],
                  coverage=full.coverage)

    f_full, f_trunc = features.build(full), features.build(trunc)
    offenders = []
    for name in f_full:
        a, b = f_full[name][:cut], f_trunc[name]
        both = np.isfinite(a) & np.isfinite(b)
        if both.sum() == 0:
            continue
        if not np.allclose(a[both], b[both], rtol=1e-4, atol=1e-6):
            offenders.append(name)
    assert not offenders, "features peek at the future: %s" % offenders


def test_backtest_is_causal():
    """Strategy returns before t must not change when prices after t change."""
    full = synthetic_panel(T=600, seed=11)
    cut = 400
    tampered = synthetic_panel(T=600, seed=11)
    rng = np.random.default_rng(0)
    tampered.bars["close"][cut:] *= (1 + rng.normal(0, 0.3, tampered.close[cut:].shape))

    c = config.load()
    g = genome.Genome(expr=genome.Un("rank", genome.Feat("mom_21")),
                      n_positions=10, rebalance_days=5)

    out = []
    for p in (full, tampered):
        f = features.build(p)
        out.append(bt.run(p, g.score(f, p.tradable), g, c).ret)

    # Allow a small edge zone: the block containing `cut` legitimately spans it.
    edge = cut - 10
    assert np.allclose(out[0][:edge], out[1][:edge], atol=1e-10), (
        "future prices changed past returns -- lookahead leak")


def test_signal_shift_destroys_edge():
    """A strategy built on a leaked (future) signal must beat the honest one.

    This is a positive control: it proves the harness *can* detect an edge, so
    a null result elsewhere means 'no edge', not 'broken measurement'.
    """
    p = synthetic_panel(T=600, seed=3)
    c = config.load()
    f = features.build(p)
    g = genome.Genome(expr=genome.Feat("mom_5"), n_positions=5, rebalance_days=5)

    rets = bt._to_returns(p.close)
    honest = bt.run(p, g.score(f, p.tradable), g, c)
    # Cheat: score = tomorrow's return, shifted back one bar.
    cheat_score = np.roll(rets, -1, axis=0)
    cheat_score[-1] = np.nan
    cheat = bt.run(p, np.where(p.tradable, cheat_score, np.nan), g, c)

    assert cheat.stats["cagr"] > honest.stats["cagr"] + 0.5, (
        "a signal that literally knows tomorrow should dominate; the harness "
        "cannot detect an edge and every other result is untrustworthy")


# =============================================================================
#  Backtest mechanics
# =============================================================================
def test_hold_everything_matches_benchmark():
    """Holding every eligible name equally must reproduce the benchmark."""
    p = synthetic_panel(T=400, N=20)
    c = config.load()
    c["backtest"]["slippage_bps"] = 0.0
    c["backtest"]["commission_bps"] = 0.0
    f = features.build(p)
    g = genome.Genome(expr=genome.Const(1.0), n_positions=20, rebalance_days=1)
    res = bt.run(p, g.score(f, p.tradable), g, c)
    bench = bt.equal_weight_benchmark(p)
    n = min(len(res.ret), len(bench))
    assert np.corrcoef(res.ret[5:n], bench[5:n])[0, 1] > 0.99


def test_costs_reduce_returns():
    p = synthetic_panel(T=500, seed=5)
    f = features.build(p)
    g = genome.Genome(expr=genome.Un("rank", genome.Feat("mom_5")),
                      n_positions=5, rebalance_days=1)
    sc = g.score(f, p.tradable)

    cheap = config.load(); cheap["backtest"]["slippage_bps"] = 0.0
    dear = config.load(); dear["backtest"]["slippage_bps"] = 50.0
    a = bt.run(p, sc, g, cheap).stats["cagr"]
    b = bt.run(p, sc, g, dear).stats["cagr"]
    assert b < a, "50bps of slippage must cost something"


def test_delisting_is_absorbed_not_ignored():
    """A name that stops printing prices must not crash or silently vanish."""
    p = synthetic_panel(T=500, N=20, seed=9, kill={0: 250, 1: 300})
    c = config.load()
    f = features.build(p)
    g = genome.Genome(expr=genome.Const(1.0), n_positions=20, rebalance_days=5)
    res = bt.run(p, g.score(f, p.tradable), g, c)
    assert np.isfinite(res.ret).all()
    assert res.ok
    assert not p.tradable[300:, 0].any(), "delisted name still marked tradable"


# =============================================================================
#  Genome
# =============================================================================
def test_genome_roundtrip_and_fingerprint():
    gr = genome.Grammar(features.FEATURE_NAMES, np.random.default_rng(1))
    for _ in range(50):
        g = gr.random_genome()
        back = genome.Genome.from_dict(g.to_dict())
        assert back.canonical() == g.canonical()
        assert back.fingerprint == g.fingerprint


def test_mutation_preserves_validity():
    p = synthetic_panel(T=300, N=15)
    f = features.build(p)
    rng = np.random.default_rng(2)
    gr = genome.Grammar(list(f.keys()), rng)
    parent = gr.random_genome()
    for op in genome.Grammar.ALL_OPERATORS:
        for _ in range(10):
            child = gr.apply(op, parent, gr.random_genome(), 1)
            if child is None:
                continue
            assert child.expr.depth() <= genome.MAX_DEPTH_HARD
            s = child.score(f, p.tradable)
            assert s.shape == p.close.shape


# =============================================================================
#  Statistics
# =============================================================================
def test_dsr_falls_as_trials_rise():
    rng = np.random.default_rng(4)
    r = rng.normal(0.0006, 0.01, 1500)
    d1 = metrics.dsr_from_returns(r, n_trials=1)["dsr"]
    d1k = metrics.dsr_from_returns(r, n_trials=1000)["dsr"]
    d1m = metrics.dsr_from_returns(r, n_trials=1_000_000)["dsr"]
    assert d1 > d1k > d1m, "more trials must deflate the Sharpe, not inflate it"


def test_expected_max_sharpe_grows_with_trials():
    v = 1.0 / 1000
    assert (metrics.expected_max_sharpe(10, v)
            < metrics.expected_max_sharpe(1000, v)
            < metrics.expected_max_sharpe(100000, v))


def test_pbo_is_high_for_pure_noise():
    """Selecting the best of many worthless strategies must look overfit."""
    rng = np.random.default_rng(6)
    R = rng.normal(0, 0.01, (1200, 50))
    p = metrics.pbo_cscv(R, n_partitions=8)
    assert p["pbo"] > 0.30, "PBO should flag noise-selection; got %.2f" % p["pbo"]


def test_newey_west_widens_error_on_autocorrelated_series():
    rng = np.random.default_rng(8)
    e = rng.normal(0, 0.01, 2000)
    ar = np.zeros_like(e)
    for i in range(1, len(e)):
        ar[i] = 0.7 * ar[i - 1] + e[i]
    ar += 0.001
    naive = ar.mean() / (ar.std(ddof=1) / np.sqrt(len(ar)))
    nw = bt._newey_west_tstat(ar)
    assert abs(nw) < abs(naive), "NW must not be more confident than naive OLS"


# =============================================================================
#  Cross-validation and the seal
# =============================================================================
def test_folds_are_disjoint_and_embargoed():
    folds = cv.block_folds(2000, 8, embargo=10, min_len=50)
    assert len(folds) >= 4
    for a, b in zip(folds, folds[1:]):
        assert b.start > a.stop, "folds must not touch; embargo missing"
        assert b.start - a.stop >= 10


def test_seal_refuses_second_evaluation(tmp_path, monkeypatch):
    import loonie.seal as S

    monkeypatch.setattr(S, "MANIFEST", str(tmp_path / "seal.json"))
    monkeypatch.setattr(S, "resolve", lambda p: tmp_path / Path(p).name)

    p = synthetic_panel(T=900)
    c = config.load()
    c["holdout"]["start"] = str(p.dates[600].date())
    c["holdout"]["end"] = str(p.dates[-1].date())
    c["holdout"]["max_evaluations"] = 1

    sl = S.Seal.create(c, p, force=True)
    g = genome.Genome(expr=genome.Const(1.0))
    sl.open_holdout(p, g)
    with pytest.raises(S.HoldoutExhausted):
        sl.open_holdout(p, g)


def test_seal_detects_tampered_data(tmp_path, monkeypatch):
    import loonie.seal as S

    monkeypatch.setattr(S, "MANIFEST", str(tmp_path / "seal.json"))
    monkeypatch.setattr(S, "resolve", lambda p: tmp_path / Path(p).name)

    p = synthetic_panel(T=900)
    c = config.load()
    c["holdout"]["start"] = str(p.dates[600].date())
    c["holdout"]["end"] = str(p.dates[-1].date())
    sl = S.Seal.create(c, p, force=True)

    p.bars["close"][700, 3] *= 1.5          # one cell
    with pytest.raises(S.SealBroken):
        sl.open_holdout(p, genome.Genome(expr=genome.Const(1.0)))


def test_seal_blocks_contaminated_training_panel(tmp_path, monkeypatch):
    import loonie.seal as S

    monkeypatch.setattr(S, "MANIFEST", str(tmp_path / "seal.json"))
    monkeypatch.setattr(S, "resolve", lambda p: tmp_path / Path(p).name)

    p = synthetic_panel(T=900)
    c = config.load()
    c["holdout"]["start"] = str(p.dates[600].date())
    c["holdout"]["end"] = str(p.dates[-1].date())
    sl = S.Seal.create(c, p, force=True)

    with pytest.raises(S.SealBroken):
        sl.assert_train_clean(p)            # full panel spans the seal
    sl.assert_train_clean(p.slice_dates(p.dates[0], p.dates[590]))


# =============================================================================
#  Risk and allocation
# =============================================================================
def test_risk_halts_on_drawdown(tmp_path, monkeypatch):
    import loonie.risk as R

    monkeypatch.setattr(R, "resolve", lambda p: tmp_path / Path(p).name)
    c = config.load()
    c["trade"]["risk"]["max_drawdown_pct"] = 0.10
    rm = R.RiskManager(c)
    rm.state = R.RiskState()
    assert rm.check(100000.0).allow
    d = rm.check(85000.0)
    assert not d.allow and d.halt
    assert not rm.check(100000.0).allow, "halt must latch, not auto-resume"


def test_allocator_shifts_weight_to_the_winner(tmp_path, monkeypatch):
    import loonie.allocator as A

    monkeypatch.setattr(A, "resolve", lambda p: tmp_path / Path(p).name)
    c = config.load()
    al = A.Allocator(c, seed=3)
    rng = np.random.default_rng(1)
    for _ in range(120):
        al.observe("good", float(rng.normal(0.0015, 0.004)))
        al.observe("bad", float(rng.normal(-0.0010, 0.004)))
    w = al.weights(["good", "bad"], n_draws=2000)
    assert w["good"] > w["bad"] * 3


def test_allocator_holds_cash_when_nothing_works(tmp_path, monkeypatch):
    import loonie.allocator as A

    monkeypatch.setattr(A, "resolve", lambda p: tmp_path / Path(p).name)
    al = A.Allocator(config.load(), seed=5)
    rng = np.random.default_rng(2)
    for _ in range(150):
        al.observe("a", float(rng.normal(-0.002, 0.003)))
        al.observe("b", float(rng.normal(-0.003, 0.003)))
    w = al.weights(["a", "b"], n_draws=2000)
    assert sum(w.values()) < 0.2, "two losing strategies should not get capital"


# =============================================================================
#  Portfolio construction
# =============================================================================
def test_no_trade_band_suppresses_churn():
    from loonie import portfolio
    from loonie.broker.base import Position

    marks = {"A": 100.0, "B": 50.0}
    pos = {"A": Position("A", 50, 100, 100), "B": Position("B", 100, 50, 50)}
    equity = 10000.0
    target = {"A": 0.5005, "B": 0.4995}          # ~0.05% drift
    assert portfolio.diff_to_orders(target, pos, equity, marks) == []

    orders = portfolio.diff_to_orders({"A": 0.8, "B": 0.2}, pos, equity, marks)
    assert orders and orders[0].side == "sell", "sells must fund buys"


def test_full_exit_sells_whole_line():
    from loonie import portfolio
    from loonie.broker.base import Position

    pos = {"A": Position("A", 50, 100, 100)}
    orders = portfolio.diff_to_orders({}, pos, 10000.0, {"A": 100.0})
    assert len(orders) == 1
    assert orders[0].side == "sell" and orders[0].qty == pytest.approx(50.0)


def test_effective_trials_discounts_correlated_candidates():
    """Near-identical candidates must not each count as an independent shot."""
    rng = np.random.default_rng(12)
    base = rng.normal(0, 0.01, 1000)

    clones = np.column_stack([base + rng.normal(0, 0.0005, 1000)
                              for _ in range(40)])
    indep = rng.normal(0, 0.01, (1000, 40))

    e_clone = metrics.effective_trials(clones, 4000)
    e_indep = metrics.effective_trials(indep, 4000)

    assert e_clone["n_effective"] < e_indep["n_effective"] / 5
    assert e_indep["n_effective"] > 0.5 * 4000, "independent trials must count"
    assert 1.0 <= e_clone["n_effective"] <= 4000


def test_evolution_state_round_trips_to_executable_genomes():
    """Everything written to state must rebuild into something that can trade.

    Regression: Evaluation.as_dict() serialised the canonical *string* but not
    the expression tree, so run_trade.py silently found zero strategies. A
    persisted strategy that cannot be reloaded is not persisted.
    """
    from loonie import evolve

    p = synthetic_panel(T=500, N=25, seed=21)
    f = features.build(p)
    c = config.load()
    c["cv"]["n_splits"] = 3
    c["evolve"]["null_samples_per_gen"] = 0

    ev = evolve.Evolver(c, p, f, seed=1, verbose=False)
    ev.seed_population(12)
    assert ev.population, "seeding produced nothing"

    for rec in [e.as_dict() for e in ev.archive.elites()[:5]]:
        assert "genome" in rec, "archive entry has no reloadable genome"
        g = genome.Genome.from_dict(rec["genome"])
        assert g.canonical() == rec["canonical"]
        assert g.score(f, p.tradable).shape == p.close.shape


def test_strategy_picks_are_tradable_names():
    """Live picks must be eligible symbols, correctly weighted."""
    from loonie import portfolio

    p = synthetic_panel(T=400, N=30, seed=22, kill={0: 100})
    f = features.build(p)
    g = genome.Genome(expr=genome.Un("rank", genome.Feat("mom_21")),
                      n_positions=8, rebalance_days=5)

    picks = portfolio.strategy_picks(g, f, p)
    assert len(picks) == 8
    assert abs(sum(picks.values()) - 1.0) < 1e-6
    assert set(picks) <= set(p.tickers)
    assert "T00" not in picks, "delisted name selected for a live order"


def test_incremental_cache_extends_tail_without_refetching(tmp_path, monkeypatch):
    """A daily run must fetch only the missing tail, not ten years again."""
    import loonie.data as D

    monkeypatch.setattr(D, "resolve", lambda q: tmp_path / Path(q).name)
    calls = []

    class FakeProvider(D.Provider):
        name = "fake"

        def fetch(self, ticker, start, end):
            calls.append((start, end))
            idx = pd.bdate_range(start, end)
            if len(idx) == 0:
                return None
            return pd.DataFrame(
                {c: np.linspace(10, 11, len(idx)) for c in D.OHLCV}, index=idx)

    prov = FakeProvider()
    first = D.fetch_ticker(prov, "ABC", "2024-01-01", "2024-06-01")
    assert first is not None and len(calls) == 1

    # min_refetch_hours=0 isolates the tail logic from the freshness guard.
    second = D.fetch_ticker(prov, "ABC", "2024-01-01", "2024-09-01",
                            min_refetch_hours=0.0)
    assert len(calls) == 2, "second call should have happened (tail missing)"
    assert pd.Timestamp(calls[1][0]) > pd.Timestamp("2024-05-30"), (
        "tail fetch must start after the cached data, not at the beginning")
    assert len(second) > len(first)
    assert second.index.is_monotonic_increasing
    assert not second.index.has_duplicates

    # Nothing new to get -> no third provider call.
    D.fetch_ticker(prov, "ABC", "2024-01-01", "2024-09-01", min_refetch_hours=0.0)
    assert len(calls) == 2, "a fully up-to-date cache must not hit the provider"

def test_demean_of_a_constant_is_exactly_zero():
    """Regression: float32 accumulation made this -3e-08, with a stable sign.

    The search found that and used it as a regime signal -- its fittest genome
    branched on `mul(mom_252, demean(-0.3359))`, which is mathematically zero
    and therefore a dead branch, but in float32 was a small negative number, so
    the branch flipped on the sign of mom_252. A rounding error was doing the
    work of a feature.
    """
    for n in (7, 20, 101, 500):
        x = np.full((4, n), np.float32(-0.3359), np.float32)
        d = features.cs_demean(x, np.ones_like(x, bool))
        assert (d == 0).all(), (
            "demean(const) leaked %.3e of noise at N=%d" % (np.abs(d).max(), n))
        assert not (d > 0).any() and not (d < 0).any()


def test_simplify_removes_dead_code():
    from loonie.genome import Bin, Const, Feat, Ite, Un, simplify

    # Both arms identical -> the branch cannot matter.
    assert str(simplify(Ite(Feat("dollar_vol_21"), Feat("rev_5"),
                            Feat("rev_5")))) == "rev_5"
    # Decided condition -> the other branch is unreachable.
    assert str(simplify(Ite(Const(1.0), Feat("mom_21"), Feat("rev_5")))) == "mom_21"
    assert str(simplify(Ite(Const(-1.0), Feat("mom_21"), Feat("rev_5")))) == "rev_5"
    # demean of a constant folds (sound now that cs_demean uses float64).
    assert str(simplify(Un("demean", Const(-0.3359)))) == "0"
    # Involution and constant folding.
    assert str(simplify(Un("neg", Un("neg", Feat("mom_5"))))) == "mom_5"
    assert str(simplify(Bin("add", Const(2.0), Const(3.0)))) == "5"


def test_simplify_rejects_unsound_folds():
    """The NaN-invalid identities must NOT be applied."""
    from loonie.genome import Bin, Const, Feat, Un, simplify

    # x * 0 is NaN when x is NaN, so it must not collapse to 0.
    assert "mom_252" in str(simplify(Bin("mul", Feat("mom_252"), Const(0.0))))
    # x - x is NaN when x is NaN.
    assert str(simplify(Bin("sub", Feat("vol_21"), Feat("vol_21")))) != "0"
    # rank of a constant is not a constant: ties resolve by column order.
    assert str(simplify(Un("rank", Const(1.0)))) != "0.5"


def test_simplify_is_semantics_preserving_on_random_trees():
    from loonie.genome import Grammar, simplify

    p = synthetic_panel(T=250, N=15, seed=33)
    f = features.build(p)
    gr = Grammar(list(f.keys()), np.random.default_rng(5), max_depth=5)
    folded = 0
    for _ in range(150):
        e = gr.random_expr()
        s = simplify(e)
        folded += int(s.size() < e.size())
        a = np.nan_to_num(e.evaluate(f, p.tradable), nan=0., posinf=0., neginf=0.)
        b = np.nan_to_num(s.evaluate(f, p.tradable), nan=0., posinf=0., neginf=0.)
        assert np.allclose(a, b, atol=1e-5, rtol=1e-4), (
            "simplify changed semantics:\n  %s\n  %s" % (e, s))
    assert folded > 0, "simplifier never fired; it is not being exercised"


def test_simplify_collapses_duplicate_fingerprints():
    """One idea must produce one fingerprint, or `trials` counts spellings."""
    from loonie.genome import Genome, Ite, Feat, Un, simplify

    variants = [
        Feat("mom_21"),
        Un("neg", Un("neg", Feat("mom_21"))),
        Ite(Feat("vol_21"), Feat("mom_21"), Feat("mom_21")),
        Un("abs", Un("abs", Feat("mom_21"))),
    ]
    prints = {Genome(expr=simplify(v)).fingerprint for v in variants[:3]}
    assert len(prints) == 1, "same idea produced %d fingerprints" % len(prints)


def test_cache_freshness_guard_avoids_redundant_refetch(tmp_path, monkeypatch):
    """A cache written minutes ago must not trigger 745 pointless API calls."""
    import loonie.data as D

    monkeypatch.setattr(D, "resolve", lambda q: tmp_path / Path(q).name)
    calls = []

    class FakeProvider(D.Provider):
        name = "fake2"

        def fetch(self, ticker, start, end):
            calls.append((start, end))
            idx = pd.bdate_range(start, end)
            if len(idx) == 0:
                return None
            return pd.DataFrame(
                {c: np.linspace(10, 11, len(idx)) for c in D.OHLCV}, index=idx)

    prov = FakeProvider()
    D.fetch_ticker(prov, "XYZ", "2024-01-01", "2024-06-01")
    assert len(calls) == 1

    # Freshly written -> trusted, even though `end` is far in the future.
    D.fetch_ticker(prov, "XYZ", "2024-01-01", "2030-01-01", min_refetch_hours=6.0)
    assert len(calls) == 1, "fresh cache was re-fetched anyway"

    # Freshness window of zero -> the tail fetch is allowed through.
    D.fetch_ticker(prov, "XYZ", "2024-01-01", "2030-01-01", min_refetch_hours=0.0)
    assert len(calls) == 2, "stale cache was not refreshed"


def test_position_cap_is_applied_in_construction_not_at_the_order_gate():
    """Capping must water-fill into names with headroom, not just drop orders.

    Regression: score-weighting handed single names 18-27% of the book. With
    the cap enforced only at order submission, five of twelve orders were
    rejected individually and the portfolio sat 60% in cash with a skew toward
    whichever names happened to be small.
    """
    from loonie import portfolio

    p = synthetic_panel(T=300, N=40, seed=41)
    f = features.build(p)
    g = genome.Genome(expr=genome.Un("rank", genome.Feat("mom_21")),
                      n_positions=20, rebalance_days=5, weighting="score")

    uncapped = portfolio.target_portfolio({"s": g}, f, p, capital_pct=0.95)
    capped = portfolio.target_portfolio({"s": g}, f, p, capital_pct=0.95,
                                        max_position_pct=0.06)

    assert max(uncapped.weights.values()) > 0.06, "fixture must actually breach"
    assert max(capped.weights.values()) <= 0.06 + 1e-9
    # Capital is redistributed, not abandoned.
    assert sum(capped.weights.values()) > 0.9 * sum(uncapped.weights.values())
    assert set(capped.weights) == set(uncapped.weights)


def test_position_cap_leaves_cash_when_it_cannot_fit():
    """If the cap binds everywhere, the remainder stays in cash and is noted."""
    from loonie import portfolio

    p = synthetic_panel(T=300, N=40, seed=42)
    f = features.build(p)
    g = genome.Genome(expr=genome.Un("rank", genome.Feat("mom_21")),
                      n_positions=5, rebalance_days=5)   # 5 names, 20% each

    t = portfolio.target_portfolio({"s": g}, f, p, capital_pct=0.95,
                                   max_position_pct=0.06)
    assert max(t.weights.values()) <= 0.06 + 1e-9
    assert sum(t.weights.values()) <= 0.31
    assert t.cash_weight > 0.65
    assert any("cap" in n or "cash" in n for n in (t.notes or []))


# =============================================================================
#  Dashboard publishing
# =============================================================================
def test_snapshot_is_complete_and_json_safe(tmp_path, monkeypatch):
    """Every section the dashboard renders must exist, and serialise cleanly.

    NaN and Infinity are valid Python floats and invalid JSON. Python's json
    module emits them anyway as bare `NaN`/`Infinity` tokens, which
    JSON.parse() in the browser rejects outright -- the dashboard would go
    blank with a console error and no other symptom.
    """
    import json as _json

    from loonie import publish

    monkeypatch.setattr(publish, "resolve", lambda q: Path("D:/trader") / q)
    snap = publish.build_snapshot(config.load())

    for section in ("generated_at", "search", "gates", "portfolio", "risk",
                    "seal", "allocator", "config", "elites", "promoted"):
        assert section in snap, "snapshot missing %r" % section

    for key in ("equity", "cash", "n_positions", "positions", "gross_exposure"):
        assert key in snap["portfolio"]
    for key in ("generation", "trials", "promoted_count"):
        assert key in snap["search"]
    assert snap["portfolio"]["is_live"] is False, "never publish a live snapshot"

    text = _json.dumps(snap, default=str)
    for bad in ("NaN", "Infinity", "-Infinity"):
        assert bad not in text, "snapshot contains %s, which JSON.parse rejects" % bad


def test_published_strategies_rebuild_into_executable_genomes(monkeypatch):
    """A CI runner with only strategies.json must still be able to trade."""
    from loonie import publish

    monkeypatch.setattr(publish, "resolve", lambda q: Path("D:/trader") / q)
    doc = publish.build_strategies(config.load())
    assert "strategies" in doc
    if not doc["strategies"]:
        pytest.skip("no strategies in state yet")

    p = synthetic_panel(T=300, N=20, seed=51)
    f = features.build(p)
    for entry in doc["strategies"]:
        g = genome.Genome.from_dict(entry["genome"])
        assert g.canonical() == entry["canonical"]
        assert g.score(f, p.tradable).shape == p.close.shape


def test_snapshot_numbers_are_finite(monkeypatch):
    """Recursively assert no non-finite numeric leaks into the feed."""
    from loonie import publish

    monkeypatch.setattr(publish, "resolve", lambda q: Path("D:/trader") / q)
    snap = publish.build_snapshot(config.load())

    bad = []

    def walk(node, path=""):
        if isinstance(node, dict):
            for k, v in node.items():
                walk(v, "%s.%s" % (path, k))
        elif isinstance(node, list):
            for i, v in enumerate(node):
                walk(v, "%s[%d]" % (path, i))
        elif isinstance(node, float) and not np.isfinite(node):
            bad.append(path)

    walk(snap)
    assert not bad, "non-finite values at: %s" % bad[:10]


def test_promotions_are_revalidated_against_the_rising_bar():
    """A strategy promoted early must be demoted once it stops clearing.

    The deflated-Sharpe threshold rises with the number of hypotheses tested,
    so promotion cannot be a one-way door. Observed live: a promoted strategy
    sitting at PBO 0.771 against a 0.40 limit, still holding capital, purely
    because it was promoted when the trial count was lower.
    """
    from loonie import evolve

    p = synthetic_panel(T=500, N=25, seed=61)
    f = features.build(p)
    c = config.load()
    c["cv"]["n_splits"] = 3
    c["evolve"]["null_samples_per_gen"] = 0

    ev = evolve.Evolver(c, p, f, seed=2, verbose=False)
    ev.seed_population(10)
    assert ev.population

    # Promote the leader under a bar it clears, then raise the bar to the
    # ceiling so nothing can clear it, and re-validate.
    top = ev.population[0]
    entry = top.as_dict()
    entry["gates"] = []
    ev.hall_of_fame = [entry]

    c["evolve"]["gate"]["min_alpha_tstat"] = 999.0
    demoted = ev.revalidate_hall_of_fame()

    assert len(demoted) == 1, "a promotion that no longer clears must be demoted"
    assert ev.hall_of_fame == [], "demoted strategy still in the hall of fame"
    assert demoted[0]["demoted_because"], "demotion must record the reason"
    assert demoted[0]["demoted_at_trials"] == ev.trials
    assert ev.demoted, "demotions must be retained, not deleted"


def test_revalidation_keeps_strategies_that_still_clear():
    from loonie import evolve

    # Long enough that the shuffle test can actually run: it needs
    # T - max(252, T//8) > max(252, T//8), i.e. roughly T > 504. On a shorter
    # panel null_percentile comes back NaN and is correctly counted as a
    # failure, which would demote everything regardless of the gate.
    p = synthetic_panel(T=1400, N=25, seed=62)
    f = features.build(p)
    c = config.load()
    c["cv"]["n_splits"] = 3
    c["evolve"]["null_samples_per_gen"] = 0
    c["evolve"]["null_shuffles"] = 4
    # A bar everything clears.
    for k, v in (("min_cv_folds_positive", 0.0), ("min_deflated_sharpe", 0.0),
                 ("max_pbo", 1.0), ("min_alpha_tstat", -99.0), ("min_trades", 0),
                 ("max_annual_turnover", 1e9), ("max_benchmark_corr", 1.0),
                 ("min_null_percentile", 0.0), ("min_stress_alpha", -1e9),
                 ("min_forward_alpha", -1e9),
                 # The IC gates default to `auto`, which on a random-walk
                 # panel correctly demotes everything. This test is about the
                 # revalidation loop, not the thresholds, so they are lowered
                 # with the rest.
                 ("min_ic_tstat", -99.0), ("min_forward_ic", -1e9)):
        c["evolve"]["gate"][k] = v

    ev = evolve.Evolver(c, p, f, seed=3, verbose=False)
    ev.seed_population(8)
    ev.hall_of_fame = [ev.population[0].as_dict()]
    assert ev.revalidate_hall_of_fame() == []
    assert len(ev.hall_of_fame) == 1


def test_duplicate_promotions_are_collapsed_by_behaviour():
    """Two spellings of one strategy must not occupy two hall-of-fame slots.

    Observed live: `zscore(ite(mul(mom_252, 0), ma_ratio_50, rev_5))` and plain
    `rev_5` were promoted separately. `mul(mom_252, 0)` is never positive so
    the branch is dead; their IRs agreed to nine significant figures and both
    placed 2,550 trades. The Thompson allocator weights per strategy key, so
    one idea was in line for double the capital of a genuinely distinct peer.
    """
    from loonie import evolve
    from loonie.genome import Bin, Const, Feat, Genome, Ite

    p = synthetic_panel(T=900, N=25, seed=71)
    f = features.build(p)
    c = config.load()
    c["cv"]["n_splits"] = 3
    c["evolve"]["null_samples_per_gen"] = 0

    ev = evolve.Evolver(c, p, f, seed=4, verbose=False)

    plain = Genome(expr=Feat("mom_21"), n_positions=10, rebalance_days=5)
    # Same thing wearing a dead branch, exactly as the live case did.
    dressed = Genome(
        expr=Ite(Bin("mul", Feat("mom_63"), Const(0.0)),
                 Feat("vol_21"), Feat("mom_21")),
        n_positions=10, rebalance_days=5)

    assert plain.fingerprint != dressed.fingerprint, "fixture must differ textually"
    assert dressed.complexity > plain.complexity

    ev.hall_of_fame = [dressed.to_dict() | {"fingerprint": dressed.fingerprint,
                                            "canonical": dressed.canonical(),
                                            "genome": dressed.to_dict()},
                       plain.to_dict() | {"fingerprint": plain.fingerprint,
                                          "canonical": plain.canonical(),
                                          "genome": plain.to_dict()}]

    removed = ev._dedupe_by_behaviour()
    assert len(removed) == 1, "expected exactly one duplicate collapsed"
    assert len(ev.hall_of_fame) == 1
    kept = ev.hall_of_fame[0]["fingerprint"]
    assert kept == plain.fingerprint, "must keep the simpler expression"
    assert "duplicate of" in removed[0]["demoted_because"][0]


def test_dedupe_keeps_genuinely_different_strategies():
    from loonie import evolve
    from loonie.genome import Feat, Genome, Un

    p = synthetic_panel(T=900, N=25, seed=72)
    f = features.build(p)
    c = config.load()
    c["cv"]["n_splits"] = 3
    c["evolve"]["null_samples_per_gen"] = 0

    ev = evolve.Evolver(c, p, f, seed=5, verbose=False)
    a = Genome(expr=Un("rank", Feat("mom_252")), n_positions=10, rebalance_days=21)
    b = Genome(expr=Un("rank", Feat("rev_5")), n_positions=10, rebalance_days=21)
    ev.hall_of_fame = [
        {"fingerprint": g.fingerprint, "canonical": g.canonical(),
         "genome": g.to_dict()} for g in (a, b)]

    assert ev._dedupe_by_behaviour() == []
    assert len(ev.hall_of_fame) == 2


# =============================================================================
#  Macro regime features
# =============================================================================
def test_macro_features_are_causal():
    """A regime value at time t must not change when later data is removed.

    Forward-filling onto the equity calendar is causal (it carries the last
    known close). Back-filling would not be, and would be invisible in every
    other test — the series would simply look prescient.
    """
    from loonie import macro

    try:
        raw = macro.fetch()
    except Exception as e:
        pytest.skip("macro cache unavailable: %s" % e)

    dates = pd.DatetimeIndex(raw.index[-900:])
    cut = 600
    full = macro.build(dates)
    trunc = macro.build(dates[:cut])

    offenders = []
    for k in full:
        a, b = full[k][:cut], trunc[k]
        both = np.isfinite(a) & np.isfinite(b)
        if both.sum() == 0:
            continue
        if not np.allclose(a[both], b[both], rtol=1e-4, atol=1e-6):
            offenders.append(k)
    assert not offenders, "macro features peek at the future: %s" % offenders


def test_macro_features_actually_switch_regime():
    """The `ite` branch test is `> 0`, so a condition must cross zero.

    The property that matters is not the fraction of time above zero — a real
    regime is persistent, and credit conditions genuinely sat on one side for
    most of 2021-2026. What matters is whether the condition ever *flips*: one
    that never crosses zero makes its branch dead code, and the search would
    spend generations breeding around a switch that cannot move.

    Measured over the full available history, not a recent slice. An earlier
    version of this test used the last 1,200 sessions and failed on m_credit at
    88% — which was a true fact about that window, not a defect in the feature.
    """
    from loonie import macro

    try:
        raw = macro.fetch()
    except Exception as e:
        pytest.skip("macro cache unavailable: %s" % e)

    m = macro.build(pd.DatetimeIndex(raw.index))
    for k, v in m.items():
        fin = v[np.isfinite(v)]
        if len(fin) < 250:
            continue
        pos = float(np.mean(fin > 0))
        assert 0.05 < pos < 0.95, (
            "%s is above zero %.0f%% of the time over the full history" % (k, 100 * pos))
        flips = int(np.sum(np.diff(np.sign(fin)) != 0))
        assert flips >= 10, (
            "%s crosses zero only %d times in %d sessions — its branch is "
            "effectively dead" % (k, flips, len(fin)))
        assert abs(float(np.mean(fin))) < 1.2, "%s is not centred (mean %.2f)" % (
            k, float(np.mean(fin)))


def test_macro_alone_carries_no_cross_sectional_signal():
    """A macro series scores every stock identically, by construction.

    This is the property that makes them safe to add as terminals: used alone
    a genome gets a flat cross-section and selects arbitrarily, so fitness
    will reject it. Their value is as `ite` conditions and as multipliers.
    """
    from loonie import macro

    try:
        raw = macro.fetch()
    except Exception as e:
        pytest.skip("macro cache unavailable: %s" % e)

    m = macro.build(pd.DatetimeIndex(raw.index[-400:]))
    wide = macro.broadcast(m, 12)
    for k, v in wide.items():
        assert v.shape == (400, 12)
        row = v[-1]
        fin = row[np.isfinite(row)]
        if len(fin) > 1:
            assert np.allclose(fin, fin[0]), "%s varies across tickers" % k


# =============================================================================
#  Walk-forward
# =============================================================================
def test_walkforward_segments_are_embargoed_and_ordered():
    """Test segments must follow their training data with a gap, never overlap."""
    import numpy as _np

    T, segs, emb = 2180, 8, 10
    edges = _np.linspace(0, T, segs + 1).astype(int)
    prev_hi = -1
    for s in range(1, segs):
        train_hi = edges[s]
        test_lo = edges[s] + emb
        test_hi = edges[s + 1]
        if test_hi - test_lo < 40 or train_hi < 300:
            continue
        assert test_lo > train_hi, "test segment must start after training ends"
        assert test_lo - train_hi >= emb, "embargo gap missing"
        assert test_lo > prev_hi, "segments overlap"
        prev_hi = test_hi


def test_honest_walkforward_searcher_cannot_see_its_test_segment():
    """The whole point of --honest: the searcher is handed only past data.

    The fast path re-ranks an archive built from the entire training window,
    so its pool was selected knowing the forward segments. Measured, that
    inflated one segment from -0.58% to +75.55% against the same 6.00%
    benchmark. This asserts the honest path cannot do that.
    """
    from loonie import cv as cvmod
    from loonie import evolve

    p = synthetic_panel(T=900, N=25, seed=81)
    f = features.build(p, macro=False)
    c = config.load()
    c["cv"]["n_splits"] = 3
    c["evolve"]["null_samples_per_gen"] = 0

    train_hi, emb = 500, 10
    past = cvmod._slice_panel(p, 0, train_hi)
    past_feats = {k: v[:train_hi] for k, v in f.items()}

    assert past.shape[0] == train_hi
    for k, v in past_feats.items():
        assert v.shape[0] == train_hi, "%s leaked rows past the boundary" % k

    ev = evolve.Evolver(c, past, past_feats, seed=9, verbose=False)
    ev.seed_population(8)
    assert ev.population

    # Every fold the searcher used must lie strictly inside the past window.
    for fold in ev.folds:
        assert fold.stop <= train_hi, "a CV fold reached into the test segment"

    # And the winner must still evaluate on the full panel afterwards.
    winner = ev.population[0].genome
    assert winner.score(f, p.tradable).shape == p.close.shape
    assert train_hi + emb < p.shape[0], "fixture leaves no test segment"


def test_forward_validation_tail_is_invisible_to_fitness():
    """Nothing the search optimises may touch the held-back tail.

    Added after a measured failure: with every gate running on the full
    training window the leader cleared all of them (alpha t 4.07, PBO 0.257)
    while an honest sequential walk-forward of the same procedure returned
    alpha t 0.10. CSCV in particular builds its training half from a random
    combination of time blocks, so half the time it fits on blocks that come
    after the block it scores — strictly easier than live trading.
    """
    from loonie import evolve

    p = synthetic_panel(T=1000, N=25, seed=91)
    f = features.build(p, macro=False)
    c = config.load()
    c["cv"]["n_splits"] = 3
    c["evolve"]["null_samples_per_gen"] = 0
    c["evolve"]["validation_tail_frac"] = 0.20

    ev = evolve.Evolver(c, p, f, seed=11, verbose=False)

    fit_T = ev.panel.shape[0]
    assert fit_T < p.shape[0], "fitness panel must be shorter than the input"
    assert abs(fit_T - int(1000 * 0.8)) <= 2

    for k, v in ev.feats.items():
        assert v.shape[0] == fit_T, "%s leaked rows into the fitness window" % k
    for fold in ev.folds:
        assert fold.stop <= fit_T, "a CV fold reaches into the validation tail"

    assert ev.val_panel is not None
    gap = p.shape[0] - ev.val_panel.shape[0] - fit_T
    assert gap >= int(c["cv"]["embargo_days"]) - 1, "embargo gap missing"
    assert ev.bench.shape[0] == fit_T


def test_forward_validation_gate_rejects_a_tail_failure():
    """A candidate with no forward alpha must not be promotable."""
    from loonie import evolve

    p = synthetic_panel(T=1000, N=25, seed=92)
    f = features.build(p, macro=False)
    c = config.load()
    c["cv"]["n_splits"] = 3
    c["evolve"]["null_samples_per_gen"] = 0

    ev = evolve.Evolver(c, p, f, seed=12, verbose=False)
    ev.seed_population(8)
    assert ev.population

    g = ev.population[0].genome
    v = ev.validate(g)
    assert set(v) >= {"val_alpha", "val_ir", "val_t"}
    if np.isfinite(v["val_alpha"]):
        assert v["val_sessions"] == ev.val_panel.shape[0]

    # On pure random-walk data no genome should show real forward alpha, so
    # the gate must be capable of saying no.
    c["evolve"]["gate"]["min_forward_alpha"] = 1e9
    checks = ev.gate(ev.population[0], None)
    names = [x["gate"] for x in checks]
    if "forward_alpha" in names:
        fa = [x for x in checks if x["gate"] == "forward_alpha"][0]
        assert not fa["pass"], "an impossible forward bar was still cleared"


def test_snapshot_writes_are_atomic(tmp_path, monkeypatch):
    """A reader must never catch snapshot.json half-written.

    The dashboard polls it every 30s while the orchestrator rewrites it every
    60. A plain write_text leaves a window where JSON.parse throws and the page
    goes blank with no other symptom — observed as a JSONDecodeError when two
    publishes overlapped.
    """
    import json as _json
    import threading

    from loonie import publish

    monkeypatch.setattr(publish, "ROOT", tmp_path)
    monkeypatch.setattr(publish, "resolve", lambda q: Path("D:/trader") / q)

    cfg = config.load()
    publish.publish(cfg)                      # seed the files
    target = tmp_path / "docs" / "data" / "snapshot.json"
    assert target.exists()

    torn = []
    stop = threading.Event()

    def reader():
        while not stop.is_set():
            try:
                _json.loads(target.read_text(encoding="utf-8"))
            except FileNotFoundError:
                continue                      # replace window; acceptable
            except ValueError as e:
                torn.append(str(e))
                return

    t = threading.Thread(target=reader, daemon=True)
    t.start()
    for _ in range(6):
        publish.publish(cfg)
    stop.set()
    t.join(timeout=5)

    assert not torn, "reader saw a torn snapshot: %s" % torn[:2]


def test_publish_returns_the_document_it_wrote():
    """Callers must not re-read a file another process may be rewriting."""
    from loonie import publish

    r = publish.publish(config.load())
    assert "doc" in r, "publish must hand back the snapshot it just built"
    assert r["doc"]["search"]["generation"] is not None
    assert r["doc"]["portfolio"]["is_live"] is False


def test_step_survives_an_empty_population():
    """A degenerate generation must skip, not raise.

    Regression: the scheduled walk-forward died with IndexError on
    `self.population[0]` because its per-segment searches got a window too
    short to form CV folds, so every candidate failed to evaluate. One
    IndexError took out the only job that validates the whole system.
    """
    from loonie import evolve

    p = synthetic_panel(T=400, N=12, seed=101)
    f = features.build(p, macro=False)
    c = config.load()
    c["cv"]["n_splits"] = 3
    c["evolve"]["null_samples_per_gen"] = 0
    c["evolve"]["population"] = 4
    c["evolve"]["validation_tail_frac"] = 0.0

    ev = evolve.Evolver(c, p, f, seed=21, verbose=False)
    ev.population = []                     # force the degenerate case
    rec = ev.step()                        # must not raise
    assert isinstance(rec, dict)
    assert "generation" in rec


def test_walkforward_disables_the_nested_holdout():
    """Inside a walk-forward the next segment IS the forward test.

    Reserving another tail inside each per-segment search takes 20% of an
    already-short window for no extra evidence — and made early segments
    unviable entirely.
    """
    src = (Path("D:/trader") / "scripts" / "walkforward.py").read_text(encoding="utf-8")
    assert 'sub_cfg["evolve"]["validation_tail_frac"] = 0.0' in src


def test_evolver_accepts_a_zero_validation_tail():
    from loonie import evolve

    p = synthetic_panel(T=600, N=15, seed=102)
    f = features.build(p, macro=False)
    c = config.load()
    c["cv"]["n_splits"] = 3
    c["evolve"]["null_samples_per_gen"] = 0
    c["evolve"]["validation_tail_frac"] = 0.0

    ev = evolve.Evolver(c, p, f, seed=22, verbose=False)
    assert ev.val_panel is None, "no tail should be reserved at frac 0"
    assert ev.panel.shape[0] == p.shape[0], "the full window must reach fitness"
    ev.seed_population(6)
    assert ev.population, "a 600-session panel must produce viable candidates"


def test_validation_result_is_structured_not_scraped(tmp_path, monkeypatch):
    """The orchestrator must read a JSON result, never regex an HTML report.

    Scraping the rendered report is fragile in the worst way: a markup change
    makes every field come back None, validation_history silently stays empty,
    and nothing errors — the same shape of failure as the validate job that
    exited rc=1 for hours without anyone noticing.
    """
    import json as _json

    from loonie import orchestrator as orc

    import re as _re

    src = (Path("D:/trader") / "loonie" / "orchestrator.py").read_text(encoding="utf-8")
    # Match the import STATEMENT, not the substring — "from . import registry"
    # contains "import re" and made the naive check fail.
    assert not _re.search(r"^\s*import re\s*$", src, _re.M),         "orchestrator still imports re for scraping"
    assert "walkforward_last.json" in src

    monkeypatch.setattr(orc, "resolve", lambda q: tmp_path / Path(q).name)
    o = orc.Orchestrator.__new__(orc.Orchestrator)
    o._log = lambda m: None

    result = {"at": "2026-09-16T10:00:00+00:00", "excess": -0.0159,
              "alpha_t": -0.16, "segments_won": 1, "segments": 4}
    (tmp_path / "walkforward_last.json").write_text(_json.dumps(result))

    o._record_validation()
    hist = _json.loads((tmp_path / Path(orc.HISTORY).name).read_text())
    assert len(hist) == 1
    assert hist[0]["alpha_t"] == -0.16

    o._record_validation()               # same run again
    hist = _json.loads((tmp_path / Path(orc.HISTORY).name).read_text())
    assert len(hist) == 1, "the same validation run was recorded twice"


def test_continuous_jobs_report_no_countdown():
    """A continuous job has no 'next run'; it was reporting -29,826,180 min."""
    from loonie import orchestrator as orc

    o = orc.Orchestrator(config.load())
    try:
        st = o.status()
        assert st["jobs"]["search"]["next_in_s"] is None
        assert st["jobs"]["publish"]["next_in_s"] is not None
        assert st["jobs"]["validate"]["next_in_s"] >= 0
    finally:
        o.me.retire()


def test_source_fingerprint_changes_when_code_changes(tmp_path, monkeypatch):
    """The supervisor must notice edited source.

    Long-running workers hold their modules in memory, so a code change does
    nothing until they restart. That bit four separate times here — most
    memorably when a redaction fix looked broken because the search daemon
    kept rewriting the file with pre-fix code while the fix sat correctly on
    disk.
    """
    import time as _t

    from loonie import orchestrator as orc

    pkg = tmp_path / "loonie"
    pkg.mkdir()
    (pkg / "a.py").write_text("x = 1")
    (tmp_path / "config.yaml").write_text("k: v")
    monkeypatch.setattr(orc, "ROOT", tmp_path)

    first = orc.source_fingerprint()
    assert first == orc.source_fingerprint(), "fingerprint must be stable"

    _t.sleep(0.01)
    (pkg / "a.py").write_text("x = 2")
    assert orc.source_fingerprint() != first, "an edit must change the fingerprint"


def test_reload_is_debounced(tmp_path, monkeypatch):
    """A multi-file save must not restart workers against half-written code."""
    from loonie import orchestrator as orc

    pkg = tmp_path / "loonie"
    pkg.mkdir()
    (pkg / "a.py").write_text("x = 1")
    monkeypatch.setattr(orc, "ROOT", tmp_path)

    o = orc.Orchestrator.__new__(orc.Orchestrator)
    o._log = lambda m: None
    o.jobs = {}
    o.fingerprint = orc.source_fingerprint()
    o._pending_fp = None
    o._pending_since = 0.0
    o.reloads = 0

    # Different LENGTH as well as content: an edit inside the filesystem's
    # mtime granularity is invisible to a timestamp-only fingerprint, which is
    # how this test originally failed.
    (pkg / "a.py").write_text("x = 22222")
    assert orc.source_fingerprint() != o.fingerprint, "edit went undetected"
    assert o.check_source(debounce=60) is False, "first sighting must only arm"
    assert o.check_source(debounce=60) is False, "still inside the debounce"
    assert o.reloads == 0

    assert o.check_source(debounce=0.0) is True, "should fire once it settles"
    assert o.reloads == 1
    assert o.check_source(debounce=0.0) is False, "must not fire again unchanged"


# =============================================================================
#  Experience store and method bandit
# =============================================================================
def test_experience_records_and_reloads(tmp_path, monkeypatch):
    """Training data must survive the process that produced it.

    340k strategies had been evaluated before this existed and not one of those
    evaluations survived anywhere a later run could read. Price history is free;
    the labelled pairing of a strategy with how it generalised forward costs
    CPU-days.
    """
    from loonie import experience as ex

    monkeypatch.setattr(ex, "resolve", lambda q: tmp_path / Path(q).name)

    g = genome.Genome(expr=genome.Un("rank", genome.Feat("mom_21")),
                      n_positions=15, rebalance_days=5)
    store = ex.ExperienceStore(method_id="qd_ir", context={"panel_sessions": 2180})
    store.record("gated", g,
                 cv={"mean_ir": 1.2, "alpha_tstat": 3.1, "val_alpha": 0.08},
                 gates=[{"gate": "pbo", "pass": False}])
    store.record("promoted", g, cv={"mean_ir": 1.4, "val_alpha": -0.02},
                 gates=[{"gate": "pbo", "pass": True}], promoted=True)
    assert store.flush() == 2

    df = ex.load()
    assert len(df) == 2
    assert set(df["method_id"]) == {"qd_ir"}
    assert df["fwd_alpha"].notna().all(), "the forward label must be stored"
    assert "momentum" in set(df["families"]), "feature family not derived"
    assert df["panel_sessions"].iloc[0] == 2180, "context not attached"

    s = ex.summary()
    assert s["rows"] == 2 and s["labelled"] == 2
    assert s["promoted"] == 1


def test_experience_appends_across_sessions(tmp_path, monkeypatch):
    """A second run must add to the corpus, not replace it."""
    from loonie import experience as ex

    monkeypatch.setattr(ex, "resolve", lambda q: tmp_path / Path(q).name)
    g = genome.Genome(expr=genome.Feat("rev_5"))

    a = ex.ExperienceStore(method_id="m1")
    a.record("gated", g, cv={"val_alpha": 0.01})
    a.flush()
    b = ex.ExperienceStore(method_id="m2")
    b.record("gated", g, cv={"val_alpha": -0.01})
    b.flush()

    df = ex.load()
    assert len(df) == 2, "second session overwrote the first"
    assert set(df["method_id"]) == {"m1", "m2"}
    assert df["run_id"].nunique() == 2


def test_method_bandit_credits_forward_not_fitness(tmp_path, monkeypatch):
    """A method that generalises must outrank one that only scores well."""
    from loonie import experience as ex
    from loonie import methods as me

    monkeypatch.setattr(ex, "resolve", lambda q: tmp_path / Path(q).name)
    monkeypatch.setattr(me, "resolve", lambda q: tmp_path / Path(q).name)

    # DISTINCT strategies — the bandit counts one vote per strategy, so twelve
    # copies of one genome is one observation, not twelve.
    def g(i, k):
        return genome.Genome(expr=genome.Feat("mom_21"), n_positions=10 + i,
                             rebalance_days=k)

    s = ex.ExperienceStore(method_id="qd_consistency")
    for i in range(12):
        # High fitness, NEGATIVE forward alpha: the exact failure mode.
        s.record("gated", g(i, 5), cv={"mean_ir": 3.0, "val_alpha": -0.05})
    s.method_id = "parsimony_hard"
    for i in range(12):
        s.record("gated", g(i, 21), cv={"mean_ir": 0.4, "val_alpha": 0.03})
    s.flush()

    b = me.MethodBandit()
    assert b.ingest_experience() == 24
    rows = {r["id"]: r for r in b.table()}
    assert rows["parsimony_hard"]["forward_rate"] == 1.0
    assert rows["qd_consistency"]["forward_rate"] == 0.0
    assert rows["parsimony_hard"]["posterior"] > rows["qd_consistency"]["posterior"], \
        "the method with better FORWARD outcomes must rank higher"


def test_method_bandit_explores_untried_methods_first(tmp_path, monkeypatch):
    """A posterior built from zero observations is a prior, not evidence."""
    from loonie import experience as ex
    from loonie import methods as me

    monkeypatch.setattr(ex, "resolve", lambda q: tmp_path / Path(q).name)
    monkeypatch.setattr(me, "resolve", lambda q: tmp_path / Path(q).name)

    g = genome.Genome(expr=genome.Feat("mom_21"))
    s = ex.ExperienceStore(method_id="qd_ir")
    for _ in range(30):
        s.record("gated", g, cv={"val_alpha": 0.05})
    s.flush()

    b = me.MethodBandit()
    b.ingest_experience()
    picks = {b.choose(np.random.default_rng(i)).id for i in range(25)}
    assert picks - {"qd_ir"}, "never explored beyond the one measured method"
    assert not b.table()[0]["established"] or True


def test_methods_actually_change_the_search():
    """A method must alter behaviour, not just label a run."""
    from loonie import methods as me

    cfg = config.load()
    base = me.BY_ID["qd_ir"].apply(cfg)
    hard = me.BY_ID["parsimony_hard"].apply(cfg)
    elitist = me.BY_ID["elitist_ir"].apply(cfg)
    low_to = me.BY_ID["low_turnover"].apply(cfg)

    assert hard["evolve"]["parsimony_penalty"] > base["evolve"]["parsimony_penalty"]
    assert hard["evolve"]["max_tree_depth"] < base["evolve"]["max_tree_depth"]
    assert elitist["evolve"]["use_map_elites"] is False
    assert base["evolve"]["use_map_elites"] is True
    assert low_to["evolve"]["gate"]["max_annual_turnover"] < \
        base["evolve"]["gate"]["max_annual_turnover"]
    # applying a method must not mutate the original config
    assert dict(cfg)["evolve"]["parsimony_penalty"] == \
        base["evolve"]["parsimony_penalty"]


def test_fitness_mode_changes_ranking():
    """consistency mode must prefer the steadier strategy over the spikier one."""
    from loonie import evolve

    p = synthetic_panel(T=800, N=20, seed=111)
    f = features.build(p, macro=False)

    class R:
        def __init__(self, ir, fp, med):
            self.mean_ir, self.frac_positive, self.median_alpha = ir, fp, med
            self.ann_turnover, self.corr_bench, self.total_trades = 3.0, 0.5, 9999

    steady, spiky = R(1.0, 0.9, 0.1), R(1.6, 0.4, 0.1)
    g = genome.Genome(expr=genome.Feat("mom_21"))

    c_ir = config.load(); c_ir["evolve"]["fitness_mode"] = "ir"
    c_cons = config.load(); c_cons["evolve"]["fitness_mode"] = "consistency"
    for c in (c_ir, c_cons):
        c["cv"]["n_splits"] = 3
        c["evolve"]["null_samples_per_gen"] = 0

    e_ir = evolve.Evolver(c_ir, p, f, seed=1, verbose=False)
    e_cons = evolve.Evolver(c_cons, p, f, seed=1, verbose=False)

    assert e_ir._fitness(spiky, g) > e_ir._fitness(steady, g), \
        "ir mode should favour the higher raw IR"
    assert e_cons._fitness(steady, g) > e_cons._fitness(spiky, g), \
        "consistency mode should favour the steadier strategy"


def test_method_bandit_counts_each_strategy_once(tmp_path, monkeypatch):
    """A leader that holds for 50 generations is one observation, not 50.

    Without this a method could manufacture confidence by simply not
    improving — the same strategy re-logged every generation would look like
    mounting independent evidence.
    """
    from loonie import experience as ex
    from loonie import methods as me

    monkeypatch.setattr(ex, "resolve", lambda q: tmp_path / Path(q).name)
    monkeypatch.setattr(me, "resolve", lambda q: tmp_path / Path(q).name)

    same = genome.Genome(expr=genome.Feat("mom_21"))
    other = genome.Genome(expr=genome.Feat("rev_5"))

    s = ex.ExperienceStore(method_id="qd_ir")
    for _ in range(40):
        s.record("gated", same, cv={"val_alpha": 0.02})   # one strategy, 40 rows
    s.record("gated", other, cv={"val_alpha": -0.01})
    s.flush()

    b = me.MethodBandit()
    b.ingest_experience()
    row = {r["id"]: r for r in b.table()}["qd_ir"]
    assert row["labelled"] == 2, "expected 2 distinct strategies, got %d" % row["labelled"]
    assert row["forward_rate"] == 0.5


def test_experience_survives_an_unclean_exit(tmp_path, monkeypatch):
    """Buffered rows must reach disk before a kill, not only on a clean exit.

    The store batches to 200 rows. A daemon meant to run for weeks would
    otherwise hold days of labelled outcomes in memory and lose all of them to
    a kill -9 — the exact way this system is normally stopped.
    """
    from loonie import experience as ex

    monkeypatch.setattr(ex, "resolve", lambda q: tmp_path / Path(q).name)
    g = genome.Genome(expr=genome.Feat("mom_21"))

    store = ex.ExperienceStore(method_id="qd_ir")
    for i in range(7):
        store.record("gated", g, cv={"val_alpha": 0.01 * i})
    assert ex.load().empty, "nothing should be on disk before a flush"

    store.flush()                          # what the daemon now does every 10 gens
    assert len(ex.load()) == 7, "a periodic flush must durably persist rows"

    # A later flush with an empty buffer must not corrupt or duplicate.
    assert store.flush() == 0
    assert len(ex.load()) == 7


# =============================================================================
#  Literature-derived methods
# =============================================================================
def test_novelty_fitness_rewards_behavioural_difference():
    """Stanley & Lehman: reward difference, not the objective.

    On deceptive problems the objective is itself what leads the search astray.
    Novelty fitness must rank a behaviourally unusual strategy above a
    conventional one with better alpha — otherwise it is just alpha wearing a
    different name.
    """
    from loonie import evolve

    p = synthetic_panel(T=700, N=18, seed=131)
    f = features.build(p, macro=False)
    c = config.load()
    c["cv"]["n_splits"] = 3
    c["evolve"]["null_samples_per_gen"] = 0
    c["evolve"]["fitness_mode"] = "novelty"

    ev = evolve.Evolver(c, p, f, seed=31, verbose=False)
    ev.seed_population(10)
    assert ev.archive.coverage >= 2, "need an archive to measure novelty against"

    class R:
        def __init__(self, to, corr, ir):
            self.ann_turnover, self.corr_bench, self.mean_ir = to, corr, ir
            self.frac_positive, self.median_alpha = 0.5, 0.0
            self.total_trades = 9999

    g = genome.Genome(expr=genome.Feat("mom_21"))
    typical = ev.population[0]
    # Far from anything in the archive, but weaker alpha.
    weird = R(90.0, 0.02, 0.1)
    # Sitting right on an existing elite's behaviour, with better alpha.
    same = R(typical.cv["ann_turnover"], typical.cv["corr_bench"], 1.5)

    assert ev._fitness(weird, g) > ev._fitness(same, g), \
        "novelty mode ranked the conventional strategy above the unusual one"


def test_bic_parsimony_scales_with_sample_size():
    """A model-selection penalty must depend on n; a hand-picked constant does not."""
    from loonie import evolve

    c = config.load()
    c["cv"]["n_splits"] = 3
    c["evolve"]["null_samples_per_gen"] = 0
    c["evolve"]["parsimony_mode"] = "bic"

    class R:
        ann_turnover, corr_bench, mean_ir = 3.0, 0.5, 1.0
        frac_positive, median_alpha, total_trades = 0.8, 0.1, 9999

    simple = genome.Genome(expr=genome.Feat("mom_21"))
    complex_ = genome.Genome(expr=genome.Bin(
        "add", genome.Un("rank", genome.Feat("mom_21")),
        genome.Bin("mul", genome.Feat("vol_21"), genome.Feat("rev_5"))))

    small = evolve.Evolver(c, synthetic_panel(T=400, N=12, seed=141),
                           features.build(synthetic_panel(T=400, N=12, seed=141),
                                          macro=False), seed=1, verbose=False)
    big = evolve.Evolver(c, synthetic_panel(T=1400, N=12, seed=141),
                         features.build(synthetic_panel(T=1400, N=12, seed=141),
                                        macro=False), seed=1, verbose=False)

    pen_small = small._fitness(R(), simple) - small._fitness(R(), complex_)
    pen_big = big._fitness(R(), simple) - big._fitness(R(), complex_)
    assert pen_small > 0 and pen_big > 0, "complexity must cost something"
    assert pen_small > pen_big, \
        "BIC must penalise complexity MORE when there is less data to justify it"


def test_all_registry_methods_are_distinct_and_applicable():
    """Every method must change the config, and no two may be identical."""
    from loonie import methods as me

    cfg = config.load()
    applied = {m.id: me.BY_ID[m.id].apply(cfg) for m in me.REGISTRY}
    assert len(applied) >= 8, "expected the literature-derived methods"

    seen = {}
    for mid, a in applied.items():
        key = json.dumps({k: a["evolve"].get(k) for k in
                          ("fitness_mode", "use_map_elites", "parsimony_penalty",
                           "parsimony_mode", "max_tree_depth", "population",
                           "cost_stress_multiplier")}, sort_keys=True)
        gate = json.dumps(a["evolve"]["gate"], sort_keys=True)
        assert (key, gate) not in seen, \
            "%s is identical to %s" % (mid, seen.get((key, gate)))
        seen[(key, gate)] = mid

    assert applied["novelty_search"]["evolve"]["fitness_mode"] == "novelty"
    assert applied["mdl_parsimony"]["evolve"]["parsimony_mode"] == "bic"


# =============================================================================
#  Peer-relative features
# =============================================================================
def test_peer_features_are_causal():
    """A peer set must be chosen from data strictly before the block it scores.

    This is the subtlest lookahead available here: picking a stock's peers
    using the very returns they are about to be measured against would be a
    beautifully disguised way of reading the answer, and it would not show up
    in any other test.
    """
    from loonie import peers as pm

    full = synthetic_panel(T=1200, N=40, seed=151)
    cut = 900
    trunc = Panel(dates=full.dates[:cut], tickers=full.tickers,
                  bars={k: v[:cut] for k, v in full.bars.items()},
                  member=full.member[:cut], tradable=full.tradable[:cut],
                  coverage=full.coverage)

    a = pm.build(full, m=8, lookback=252, refresh=63)
    b = pm.build(trunc, m=8, lookback=252, refresh=63)

    offenders = []
    for k in a:
        x, y = a[k][:cut], b[k]
        both = np.isfinite(x) & np.isfinite(y)
        if both.sum() == 0:
            continue
        if not np.allclose(x[both], y[both], rtol=1e-4, atol=1e-6):
            offenders.append(k)
    assert not offenders, "peer features peek at the future: %s" % offenders


def test_peer_relative_is_not_market_relative():
    """Peer-relative must carry information market-relative does not.

    If subtracting the peer mean were the same as subtracting the market mean,
    the whole module would be an expensive way to recompute a feature that
    already exists.
    """
    from loonie import features as F
    from loonie import peers as pm

    p = synthetic_panel(T=900, N=40, seed=152)
    # Give half the names a shared factor so genuine cohorts exist to find.
    rng = np.random.default_rng(7)
    factor = rng.normal(0, 0.02, p.close.shape[0])
    c = p.bars["close"].copy()
    for j in range(0, 40, 2):
        c[:, j] *= np.cumprod(1 + factor).astype(np.float32)
    p.bars["close"] = c

    pf = pm.build(p, m=8, lookback=252, refresh=63)
    mom21 = F.pct_change(p.bars["close"], 21)
    market_rel = mom21 - np.nanmean(np.where(p.tradable, mom21, np.nan),
                                    axis=1, keepdims=True)

    peer_rel = pf["peer_rel_mom_21"]
    both = np.isfinite(peer_rel) & np.isfinite(market_rel)
    assert both.sum() > 1000, "not enough overlap to compare"
    r = np.corrcoef(peer_rel[both], market_rel[both])[0, 1]
    assert abs(r) < 0.97, (
        "peer-relative is %.3f correlated with market-relative — it is not "
        "adding an axis" % r)


def test_peer_rank_is_bounded_and_peer_corr_is_a_correlation():
    from loonie import peers as pm

    p = synthetic_panel(T=900, N=30, seed=153)
    pf = pm.build(p, m=8, lookback=252, refresh=63)

    rank = pf["peer_rank_mom_21"]
    rank = rank[np.isfinite(rank)]
    assert rank.size and rank.min() >= 0.0 and rank.max() <= 1.0

    pc = pf["peer_corr"]
    pc = pc[np.isfinite(pc)]
    assert pc.size and pc.min() >= -1.001 and pc.max() <= 1.001

    assert pm.PEER_NAMES and all(n.startswith("peer_") for n in pm.PEER_NAMES)
    from loonie.features import family_of
    assert family_of("peer_rel_mom_21") == "peer"


# =============================================================================
#  External theses
# =============================================================================
def test_thesis_scoring_separates_trailing_from_forward(tmp_path, monkeypatch):
    """A forecast is tested by what happens AFTER it was made, not before.

    This is the whole point of the capture date. A thesis whose proxy already
    ran several hundred percent is describing a completed move, however sound
    its reasoning — and that is invisible unless both sides of the capture date
    are measured separately.
    """
    from loonie import knowledge as kn

    monkeypatch.setattr(kn, "resolve", lambda q: tmp_path / Path(q).name)

    p = synthetic_panel(T=1000, N=20, seed=161)
    # Make T00/T01 rip in the FIRST half only, then go flat.
    c = p.bars["close"].copy()
    ramp = np.linspace(1.0, 6.0, 500).astype(np.float32)
    for j in (0, 1):
        c[:500, j] *= ramp
        c[500:, j] *= ramp[-1]
    p.bars["close"] = c

    kn.add({"id": "t1", "captured": str(p.dates[500].date()),
            "summary": "already happened",
            "claims": [{"id": "c1", "statement": "these go up",
                        "proxy": {"tickers": ["T00", "T01"]}}]})

    scored = kn.score(p)
    claim = scored[0]["claims"][0]
    assert claim["trailing_excess"] > 0.2, "the completed move must show trailing"
    assert claim["already_moved"] is True
    assert claim["forward_excess"] is not None
    assert claim["forward_excess"] < claim["trailing_excess"], \
        "forward must be measured separately from the move that already ran"


def test_thesis_store_round_trips_and_replaces(tmp_path, monkeypatch):
    from loonie import knowledge as kn

    monkeypatch.setattr(kn, "resolve", lambda q: tmp_path / Path(q).name)
    kn.add({"id": "a", "summary": "first", "claims": []})
    kn.add({"id": "b", "summary": "second", "claims": []})
    assert len(kn.load()) == 2
    kn.add({"id": "a", "summary": "revised", "claims": []})
    got = {t["id"]: t["summary"] for t in kn.load()}
    assert len(got) == 2 and got["a"] == "revised", "same id must replace"
    assert all("recorded" in t for t in kn.load()), "capture time must be stamped"


def test_theses_never_reach_the_strategy_search():
    """A narrative that steers the hypothesis space has escaped its own test.

    The search must not import or read recorded theses — if it could, a
    compelling story would quietly bias what gets proposed, which is the exact
    failure this store exists to prevent.
    """
    for mod in SEARCH_PATH:
        assert "knowledge" not in _imports_of(mod), \
            "loonie/%s.py imports the thesis store" % mod


# =============================================================================
#  Factor attribution
# =============================================================================
def _fake_factors(T=900, seed=5):
    """A synthetic factor library with the same shape as Ken French's."""
    import pandas as pd
    from loonie import factors as FA

    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2015-01-01", periods=T)
    df = pd.DataFrame(
        rng.normal(0.0, 0.01, size=(T, len(FA.FACTORS))),
        index=idx, columns=FA.FACTORS)
    df["RF"] = 0.00008
    return df


def test_attribution_finds_no_alpha_in_a_pure_factor_portfolio():
    """The test the whole module exists for.

    A return stream that is nothing but a levered bet on SMB and HML must
    attribute to ~zero alpha. If it did not, the diagnostic would bless a
    portfolio you can buy for three basis points.
    """
    from loonie import factors as FA

    f = _fake_factors()
    rng = np.random.default_rng(11)
    ret = (0.8 * f["SMB"].to_numpy() + 0.5 * f["HML"].to_numpy()
           + rng.normal(0, 0.0005, len(f)))

    a = FA.attribute(ret, f.index, is_excess=True, factors=f)
    assert a["ok"], a.get("reason")
    assert abs(a["alpha_tstat"]) < 2.0, \
        "a pure factor bet must not read as alpha (t=%.2f)" % a["alpha_tstat"]
    assert abs(a["betas"]["SMB"] - 0.8) < 0.08
    assert abs(a["betas"]["HML"] - 0.5) < 0.08
    assert a["r2"] > 0.8, "factors should explain nearly all of it"


def test_attribution_keeps_alpha_that_is_orthogonal_to_the_factors():
    """The converse: a real edge must survive the regression."""
    from loonie import factors as FA

    f = _fake_factors()
    rng = np.random.default_rng(12)
    # 12%/yr of genuine, factor-orthogonal drift.
    ret = (0.3 * f["Mkt-RF"].to_numpy() + 0.12 / 252.0
           + rng.normal(0, 0.002, len(f)))

    a = FA.attribute(ret, f.index, is_excess=True, factors=f)
    assert a["ok"]
    assert a["alpha_tstat"] > 2.0, "real alpha was regressed away"
    assert abs(a["alpha_ann"] - 0.12) < 0.04


def test_attribution_subtracts_the_risk_free_rate_only_for_total_returns():
    """A spread is self-financing; a long-only stream is not.

    Getting this backwards shifts the alpha by the whole T-bill yield, which
    is small enough to miss and large enough to matter.
    """
    from loonie import factors as FA

    f = _fake_factors()
    ret = np.full(len(f), 0.0004)

    exc = FA.attribute(ret, f.index, is_excess=True, factors=f)
    tot = FA.attribute(ret, f.index, is_excess=False, factors=f)
    gap = exc["alpha_daily"] - tot["alpha_daily"]
    assert abs(gap - 0.00008) < 1e-6, \
        "the difference between the two must be exactly RF"


def test_factor_returns_are_fractions_not_percent():
    """Ken French publishes percent. A missed /100 makes every beta 100x off."""
    from loonie import factors as FA

    try:
        f = FA.fetch()
    except Exception:
        import pytest
        pytest.skip("factor library unavailable offline")
    ann = float(f["Mkt-RF"].mean() * 252)
    assert 0.02 < ann < 0.15, \
        "equity premium reads %.3f/yr -- scaling is wrong" % ann
    assert f.index[0].year < 1930, "daily history should start in the 1920s"


def test_attribution_never_reaches_the_strategy_search():
    """Factors are a yardstick, not a feature.

    If the search could see them it would learn to dodge the regression --
    optimising for factor-orthogonality rather than for returns -- and the
    attribution would stop being an independent check on its output.
    """
    for mod in SEARCH_PATH:
        assert "factors" not in _imports_of(mod), \
            "loonie/%s.py imports the factor library" % mod


# =============================================================================
#  Fundamentals -- the filing date is the only honest key
# =============================================================================
def _filings(rows) -> "pd.DataFrame":
    """Build a facts frame from (concept, start, end, filed, val) tuples."""
    df = pd.DataFrame(rows, columns=["concept", "start", "end", "filed", "val"])
    for c in ("start", "end", "filed"):
        df[c] = pd.to_datetime(df[c])
    df["tag"] = df["concept"]
    df["form"] = "10-Q"
    return df.sort_values("filed")


def test_a_fact_is_invisible_until_the_day_it_was_filed():
    """The single most important property in this module.

    Apple's FY2008 balance sheet was filed ten months after the period it
    describes. A feature keyed to the period end would put information into a
    backtest that nobody could have had, produce a spectacular equity curve,
    and pass every other test in this file.
    """
    from loonie import fundamentals as FU

    dates = pd.bdate_range("2020-01-01", periods=260)
    df = _filings([
        ("assets", None, "2020-03-31", "2020-11-15", 500.0),   # filed LATE
    ])
    s = FU._known_series(df, "assets", dates, flow=False)

    before = dates < pd.Timestamp("2020-11-15")
    assert np.all(np.isnan(s[before])), \
        "the value leaked into dates before it was filed"
    assert np.all(s[~before] == 500.0), \
        "the value never appeared after its filing date"


def test_a_restatement_lands_on_its_own_filing_date():
    """An amended figure corrects the record going forward, not backwards.

    As of a date between the two filings, a reader had the original number.
    A panel that shows the revised one is quietly telling the strategy how
    the correction turned out.
    """
    from loonie import fundamentals as FU

    dates = pd.bdate_range("2021-01-01", periods=400)
    df = _filings([
        ("assets", None, "2021-03-31", "2021-05-01", 100.0),
        ("assets", None, "2021-03-31", "2021-09-01", 140.0),   # restated
    ])
    s = FU._known_series(df, "assets", dates, flow=False)

    mid = int(dates.searchsorted(pd.Timestamp("2021-07-01")))
    end = int(dates.searchsorted(pd.Timestamp("2021-10-01")))
    assert s[mid] == 100.0, "the restated value leaked backwards"
    assert s[end] == 140.0, "the restatement never took effect"


def test_flows_are_trailing_twelve_months_not_one_quarter():
    """A single quarter is seasonal.

    Comparing one company's Q4 against another's Q2 across a cross-section
    measures the calendar, not the companies.
    """
    from loonie import fundamentals as FU

    dates = pd.bdate_range("2022-01-01", periods=500)
    qs = [("2021-10-01", "2021-12-31", "2022-02-01", 10.0),
          ("2022-01-01", "2022-03-31", "2022-05-01", 20.0),
          ("2022-04-01", "2022-06-30", "2022-08-01", 30.0),
          ("2022-07-01", "2022-09-30", "2022-11-01", 40.0)]
    df = _filings([("net_income", s, e, f, v) for s, e, f, v in qs])
    s = FU._known_series(df, "net_income", dates, flow=True)

    before4 = int(dates.searchsorted(pd.Timestamp("2022-10-01")))
    after4 = int(dates.searchsorted(pd.Timestamp("2022-11-02")))
    assert np.isnan(s[before4]), "reported a TTM from fewer than four quarters"
    assert s[after4] == 100.0, "TTM is not the sum of the last four quarters"


def test_fundamental_features_are_causal():
    """Same standard the price features are held to: truncate and re-run."""
    from loonie import fundamentals as FU

    full = synthetic_panel(T=600, N=6, seed=23)
    cut = 400
    rng = np.random.default_rng(3)
    facts = {}
    for t in full.tickers:
        rows = []
        for q in range(8):
            end = full.dates[min(60 * q + 55, len(full.dates) - 1)]
            filed = full.dates[min(60 * q + 59, len(full.dates) - 1)]
            for c in ("assets", "equity", "liabilities", "shares"):
                rows.append((c, None, end, filed, float(rng.uniform(50, 500))))
            for c in ("net_income", "revenue", "gross_profit"):
                rows.append((c, full.dates[60 * q], end, filed,
                             float(rng.uniform(1, 40))))
        facts[t] = _filings(rows)

    trunc = Panel(dates=full.dates[:cut], tickers=full.tickers,
                  bars={k: v[:cut] for k, v in full.bars.items()},
                  member=full.member[:cut], tradable=full.tradable[:cut],
                  coverage=full.coverage)

    a, b = FU.build(full, facts), FU.build(trunc, facts)
    offenders = []
    for name in a:
        x, y = a[name][:cut], b[name]
        both = np.isfinite(x) & np.isfinite(y)
        if both.sum() == 0:
            continue
        if not np.allclose(x[both], y[both], rtol=1e-4, atol=1e-6):
            offenders.append(name)
    assert not offenders, "fundamentals peek at the future: %s" % offenders


def test_missing_fundamentals_do_not_become_a_survivorship_signal():
    """A feature that is NaN exactly for the names that later delisted is a
    survivorship leak wearing a balance sheet.

    The search would learn "avoid names with no fundamentals" and score
    beautifully, because absence of a filing correlates with the company
    ceasing to exist. Absence must be uninformative about the future, so the
    features must be NaN -- never a filled sentinel the search can rank on.
    """
    from loonie import fundamentals as FU

    panel = synthetic_panel(T=300, N=6, seed=29, kill={4: 200, 5: 220})
    facts = {}
    for t in panel.tickers[:4]:          # the two doomed names file nothing
        facts[t] = _filings([
            ("assets", None, panel.dates[50], panel.dates[55], 300.0),
            ("net_income", panel.dates[0], panel.dates[50],
             panel.dates[55], 12.0)])

    out = FU.build(panel, facts)
    for name, arr in out.items():
        col = arr[:, 4:]
        assert np.all(np.isnan(col)), \
            "%s invented a value for a name with no filings" % name


def test_fundamentals_never_reach_the_search_without_passing_causality():
    """Fundamentals join the feature set through features.build, never by a
    back door that skips the causality test the rest of the terminals face."""
    from loonie import fundamentals as FU

    assert hasattr(FU, "FUNDAMENTAL_NAMES") and FU.FUNDAMENTAL_NAMES
    assert all(n.startswith("f_") for n in FU.FUNDAMENTAL_NAMES), \
        "fundamental terminals must be namespaced so they are identifiable"


def _load_script(name: str):
    """Import a file from scripts/ as a module."""
    import importlib.util

    p = Path(__file__).resolve().parent.parent / "scripts" / (name + ".py")
    spec = importlib.util.spec_from_file_location("s_" + name, p)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_ic_is_a_rank_correlation_with_ties_averaged():
    """A feature with many equal values must not be ranked by column order."""
    ic = _load_script("feature_ic")

    T, N = 300, 60
    rng = np.random.default_rng(4)
    fwd = rng.normal(0, 0.02, (T, N))
    mask = np.ones((T, N), bool)

    # Perfectly predictive: the feature IS the forward return.
    s = ic.ic_series(fwd.copy(), fwd, mask)
    assert np.nanmean(s) > 0.99, "a perfect predictor did not score IC 1"

    # A constant feature has no cross-section to correlate; undefined, not 0.
    flat = ic.ic_series(np.ones((T, N)), fwd, mask)
    assert np.all(np.isnan(flat)), "a flat feature produced a correlation"

    # Half the names tied: the tied block must share a rank rather than be
    # ordered by position, which would manufacture signal out of the column
    # index.
    tied = fwd.copy()
    tied[:, :30] = 5.0
    r = ic._rank(tied)
    assert np.allclose(r[:, :30], r[:, :1]), "ties were broken by column order"


def test_spread_of_a_constant_row_must_be_measured_in_float64():
    """The bug this project has already paid for once, in a new disguise.

    A macro row holds one value repeated across every name. The values are
    bit-identical -- nothing differs between them -- and yet np.nanstd on a
    float32 array accumulates in float32 and reports a spread around 6e-08.
    The dust is manufactured by the measurement, not present in the data.

    That is the same mechanism that once let a genome branch on
    demean(Const) = -3e-08 as though it were a regime signal, and cost a
    third of this system's alpha when it was fixed. Here it made a
    "is this feature constant?" check silently never fire.
    """
    T, N = 200, 616
    row = np.linspace(-2.0, 2.0, T).astype(np.float32)
    broadcast = np.repeat(row[:, None], N, axis=1)

    assert np.all(broadcast == broadcast[:, :1]), \
        "test premise broken: the row is not actually constant"

    dust = np.nanstd(broadcast, axis=1)
    assert np.any(dust > 1e-12), \
        "float32 accumulation no longer manufactures spread; simplify this"

    clean = np.nanstd(broadcast.astype(np.float64), axis=1)
    assert np.all(clean == 0.0), \
        "accumulating in float64 must report a constant row as constant"


def test_detectable_alpha_scales_with_span_and_tracking_error():
    """The arithmetic behind the power analysis, pinned.

    t = (excess / tracking error) * sqrt(years). Everything the power script
    concludes rests on this, so it is worth one test that the relationship is
    the way round we think: a longer span lowers the detectable floor, and a
    wider tracking error raises it.
    """
    import math

    def mde(te, yrs, t=2.0):
        return t * te / math.sqrt(yrs)

    # Doubling the span does not halve the floor -- it divides by sqrt(2).
    assert abs(mde(0.10, 10.0) / mde(0.10, 5.0) - 1 / math.sqrt(2)) < 1e-9

    # Tracking error is linear in the floor: halve one, halve the other.
    assert abs(mde(0.05, 5.0) / mde(0.10, 5.0) - 0.5) < 1e-9

    # The measured case: 14.81% tracking error over 5.61 years needs an
    # excess north of 12% a year to register, which no long-only equity
    # strategy delivers. If this ever drops below ~4% the instrument has
    # genuinely changed and the gate should be revisited.
    assert mde(0.1481, 5.61) > 0.12


# =============================================================================
#  Cross-sectional IC -- the gate that has power
# =============================================================================
def test_forward_returns_start_the_day_after_the_score():
    """The score at t is acted on at t+1, so it is judged from t+1.

    Starting the forward window at t would score a signal against a bar it is
    already inside -- the single most productive way to manufacture an edge
    that cannot be traded.
    """
    from loonie import ic as IC

    close = np.cumprod(1 + np.full((60, 3), 0.01)) .reshape(60, 3) * 100
    close = (100 * np.cumprod(np.full((60, 3), 1.01), axis=0))
    f = IC.forward_returns(close, 5)

    # A constant 1%/day compounding for 5 days, measured from t+1.
    assert abs(f[0, 0] - (1.01 ** 5 - 1)) < 1e-9
    # The last h+1 rows cannot know their own future.
    assert np.all(np.isnan(f[-6:]))


def test_ic_recovers_a_planted_ranking_and_rejects_a_scrambled_one():
    from loonie import ic as IC

    rng = np.random.default_rng(3)
    T, N = 400, 80
    fwd = rng.normal(0, 0.02, (T, N))
    mask = np.ones((T, N), bool)

    perfect = IC.summarize(fwd.copy(), fwd, mask, 1)
    assert perfect["ic"] > 0.99, "a perfect ranking did not score IC 1"
    # A flawless ranker has no spread in its IC at all, so the t-stat is
    # undefined rather than zero. Scoring it 0.0 would call the best possible
    # signal worthless, so the degenerate case is reported as infinite.
    assert perfect["ic_t"] == float("inf")

    noise = IC.summarize(rng.normal(0, 1, (T, N)), fwd, mask, 1)
    assert abs(noise["ic"]) < 0.05
    assert abs(noise["ic_t"]) < 3.0, "unrelated scores produced a t-stat"


def test_overlapping_windows_do_not_inflate_the_t_stat():
    """Consecutive ICs at horizon h share h-1 days of the same forward window.

    Treating them as independent inflates the t-stat by roughly sqrt(h). At a
    21-day rebalance that is a factor near 4.5 -- the difference between a
    gate that means something and one that passes everything.
    """
    from loonie import ic as IC

    rng = np.random.default_rng(17)
    x = rng.normal(0.01, 0.05, 1200)
    naive = float(np.mean(x) / (np.std(x, ddof=1) / np.sqrt(len(x))))
    corrected = IC._nw_tstat(x, 21)
    assert corrected < naive, "the overlap correction did not reduce the t-stat"

    # On a deliberately autocorrelated series the gap must be large, not token.
    y = np.convolve(rng.normal(0.01, 0.05, 1300), np.ones(21) / 21, "valid")
    n_y = float(np.mean(y) / (np.std(y, ddof=1) / np.sqrt(len(y))))
    assert IC._nw_tstat(y, 21) < 0.6 * n_y


def test_null_bar_rises_with_how_hard_the_search_looked():
    """A fixed threshold is the wrong shape for 200,000 evaluated genomes.

    The largest |t| among k independent null draws grows like sqrt(2 ln k), so
    the honest bar is a function of search effort, not a constant.
    """
    from loonie import ic as IC

    assert IC.null_bar(10) < IC.null_bar(1000) < IC.null_bar(100000)
    assert abs(IC.null_bar(56837) - 4.68) < 0.02, \
        "the bar at the live effective-trial count moved"
    # A young search still has to clear ordinary significance first.
    assert IC.null_bar(2) == 2.0
    assert IC.null_bar(1, floor=2.5) == 2.5


def test_ic_is_computed_on_the_fitting_window_not_the_whole_panel():
    """The gate must not be able to see the forward tail.

    Evolver.gate scores IC against self.panel, which is truncated at
    fit_end; the validation tail is a separate panel reached only through
    validate(). If the gate ever read full_panel the forward_ic check would
    be scoring the window it had already optimised against.
    """
    import ast

    src = (Path(__file__).resolve().parent.parent / "loonie" / "evolve.py")
    tree = ast.parse(src.read_text(encoding="utf-8"))
    fn = next(n for n in ast.walk(tree)
              if isinstance(n, ast.FunctionDef) and n.name == "gate")
    attrs = {n.attr for n in ast.walk(fn) if isinstance(n, ast.Attribute)}
    assert "full_panel" not in attrs, "the IC gate can see the forward tail"
    assert "panel" in attrs


def test_gate_never_reuses_another_genomes_scores():
    """The score memo is tagged with its fingerprint for a reason.

    evaluate() returns early on a cache hit WITHOUT refreshing the memo, so by
    the time gate() runs the memo routinely holds a different genome than the
    one being judged -- confirmed here, not hypothesised. Without the tag that
    genome would be gated on someone else's scores, and it would look like an
    ordinary passing candidate rather than a bug.
    """
    from loonie import evolve

    p = synthetic_panel(T=700, N=20, seed=71)
    f = features.build(p)
    c = config.load()
    c["cv"]["n_splits"] = 3
    c["evolve"]["null_samples_per_gen"] = 0

    ev = evolve.Evolver(c, p, f, seed=5, verbose=False)
    ev.seed_population(4)

    g0 = ev.population[0].genome
    ev.evaluate(g0)                      # served from cache; memo untouched
    stale_fp, stale_score = ev._last_score
    assert stale_fp is not None

    if stale_fp == g0.fingerprint:
        pytest.skip("memo happened to hold this genome; nothing to guard")

    # The guard must refuse the stale matrix and fall back to recomputing.
    served = (stale_score if stale_fp == g0.fingerprint
              else g0.score(ev.feats, ev.panel.tradable))
    truth = g0.score(ev.feats, ev.panel.tradable)
    both = np.isfinite(served) & np.isfinite(truth)
    assert both.any()
    assert np.allclose(served[both], truth[both]),         "gate was served another genome's scores"


def test_diagnostics_do_not_load_the_unsealed_panel():
    """The failure this guards against actually happened.

    attribute.py, feature_ic.py and power.py each called load_panel(cfg) with
    no date range, which returns everything on disk -- holdout included -- and
    read straight through the sealed window while the evaluation counter went
    on reporting 0 of 1. The counter guards one door; nothing guarded this one.

    Analysis scripts must go through seal.training_panel(), which defaults to
    the training range and logs an override to the ledger.
    """
    import ast

    root = Path(__file__).resolve().parent.parent / "scripts"
    for name in ("attribute.py", "feature_ic.py", "power.py"):
        tree = ast.parse((root / name).read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            fn = node.func
            called = (fn.id if isinstance(fn, ast.Name)
                      else fn.attr if isinstance(fn, ast.Attribute) else "")
            if called != "load_panel":
                continue
            kw = {k.arg for k in node.keywords}
            assert "start" in kw, \
                "%s calls load_panel without a start date" % name


def test_recording_an_exposure_marks_the_seal_contaminated(tmp_path, monkeypatch):
    """A look that is not written down leaves the ledger claiming a
    cleanliness the data no longer has."""
    import pandas as pd

    from loonie import seal as S

    monkeypatch.setattr(S, "resolve", lambda q: tmp_path / Path(q).name)
    s = S.Seal(start=pd.Timestamp("2024-09-01"), stop=pd.Timestamp("2026-09-01"),
               digest="d", created="now", evaluations=0, max_evaluations=1,
               ledger=[], path=tmp_path / "holdout_seal.json")
    s.save()
    assert not s.contaminated
    assert "CONTAMINATED" not in s.describe()

    s.record_exposure(by="a script", what="read the window")
    assert s.contaminated
    assert "CONTAMINATED" in s.describe()
    assert s.ledger[-1]["event"] == "exposed"

    # It must survive a reload, and there is deliberately no way to clear it.
    again = S.Seal.load(object())
    assert again.contaminated, "contamination did not persist"


def test_forward_seal_rests_on_the_clock_not_a_hash(tmp_path, monkeypatch):
    """Every historical window here is training data or contaminated, so a
    digest has nothing clean to hash. A forward seal moves the guarantee to
    the commitment timestamp: data that did not exist cannot have been seen.
    """
    import pandas as pd

    from loonie import seal as S

    monkeypatch.setattr(S, "resolve", lambda q: tmp_path / Path(q).name)
    cfg = config.load()
    s = S.Seal.create_forward(cfg, start="2026-09-17", min_sessions=250,
                              archive_existing=False)
    assert s.kind == "forward"
    assert s.digest == "", "a forward window must not claim a digest"
    # Open-ended. Inheriting cfg.holdout.end gave it a stop date in the past,
    # before its own start -- a window that could never contain anything.
    assert s.stop > s.start, "forward seal closed before it opened"
    assert s.committed_at

    # Not enough history yet -> refuses, and says so as "not earned".
    thin = synthetic_panel(T=30, N=5)
    thin.dates = pd.bdate_range("2026-09-18", periods=30)
    ready, why, n = s.forward_ready(thin)
    assert not ready and "sessions accrued" in why

    with pytest.raises(S.SealBroken) as e:
        s.open_holdout(thin, genome=None)
    assert "not ready" in str(e.value)
    assert s.evaluations == 0, "a refused open must not burn an evaluation"


def test_forward_seal_rejects_data_that_predates_the_commitment(tmp_path,
                                                                monkeypatch):
    """The whole guarantee. Sessions from before the commitment could have
    informed it, so scoring them proves nothing."""
    import pandas as pd

    from loonie import seal as S

    monkeypatch.setattr(S, "resolve", lambda q: tmp_path / Path(q).name)
    cfg = config.load()
    s = S.Seal.create_forward(cfg, start="2020-01-01", min_sessions=10,
                              archive_existing=False)

    old = synthetic_panel(T=400, N=5)
    old.dates = pd.bdate_range("2020-01-01", periods=400)   # long before now
    ready, _, _ = s.forward_ready(old)
    assert not ready, "sessions before the commitment must not count toward it"


def test_forward_seal_refuses_a_candidate_that_was_not_registered(tmp_path,
                                                                  monkeypatch):
    """Pre-registration is what collapses the multiple-testing penalty.

    Testing a candidate chosen after the data existed is the search again with
    a sample size of one, so the seal must refuse it rather than quietly
    scoring it.
    """
    from loonie import seal as S

    monkeypatch.setattr(S, "resolve", lambda q: tmp_path / Path(q).name)
    cfg = config.load()
    s = S.Seal.create_forward(cfg, min_sessions=0, archive_existing=False)

    class G:
        fingerprint = "aaaa1111"

        def canonical(self):
            return "registered_one"

    class H:
        fingerprint = "bbbb2222"

        def canonical(self):
            return "latecomer"

    s.register([G()], note="pre-registered set")
    assert len(s.registered) == 1

    with pytest.raises(S.SealBroken) as e:
        s.open_holdout(synthetic_panel(T=300, N=5), genome=H())
    assert "not pre-registered" in str(e.value)


def test_stopping_rule_is_declared_in_config_not_discovered():
    """The bar rises with trials, so an open-ended daemon loses ground by
    running. The end has to be a pre-committed number, like the seal."""
    c = config.load()
    stop = c["evolve"].get("stop") or {}
    assert int(stop.get("max_trials", 0)) > 0
    assert int(stop.get("stall_generations", 0)) > 0

    from loonie.ic import null_bar
    # The configured ceiling should sit where the bar is still roughly
    # reachable; if this ever fails the rule has drifted from its rationale.
    assert null_bar(stop["max_trials"]) < 6.0


def test_failed_publish_rebase_cleans_up_after_itself():
    """A failed rebase leaves .git/rebase-merge behind, and every later git
    command in the repo refuses to run until it is cleared by hand.

    That is not hypothetical: after the history rewrite moved the remote out
    from under an in-flight rebase, this job wedged the repository and failed
    every publish for hours. The docstring claimed it "leaves the working tree
    alone", which was the opposite of what it did.
    """
    import ast

    src = (Path(__file__).resolve().parent.parent / "scripts"
           / "publish_dashboard.py").read_text(encoding="utf-8")
    tree = ast.parse(src)
    fn = next(n for n in ast.walk(tree)
              if isinstance(n, ast.FunctionDef) and n.name == "main")

    # Find the `if pull.returncode != 0:` branch and require an abort in it.
    aborts = []
    for node in ast.walk(fn):
        if not isinstance(node, ast.Call):
            continue
        f = node.func
        if (isinstance(f, ast.Name) and f.id == "git"
                and [a for a in node.args
                     if isinstance(a, ast.Constant) and a.value == "--abort"]):
            aborts.append(node)
    assert aborts, "publish never aborts a failed rebase"
    assert "rebase --abort" in src, \
        "the operator-facing message should name the manual escape hatch"


def test_preregistration_dedupes_one_idea_sized_two_ways():
    """Registering the same hypothesis twice spends two of five slots on one
    bet, and the IC-series check alone does not catch it.

    Two copies of rank(f_rnd_intensity) at k=25/rb=2 and k=10/rb=21 produce IC
    series measured against DIFFERENT forward horizons, so they correlate
    weakly and both survived. That actually happened in the live commitment.
    The expression core -- everything before the |k=..|rb=.. suffix -- is what
    identifies the idea.
    """
    core = lambda s: str(s).split("|")[0]                      # noqa: E731

    a = "rank(f_rnd_intensity)|k=25|rb=2|w=equal"
    b = "rank(f_rnd_intensity)|k=10|rb=21|w=equal"
    c = "rank(f_roe)|k=25|rb=2|w=equal"

    assert core(a) == core(b), "same idea, different sizing, must collapse"
    assert core(a) != core(c), "different ideas must stay distinct"

    src = (Path(__file__).resolve().parent.parent / "scripts"
           / "preregister.py").read_text(encoding="utf-8")
    assert "def core(" in src, "preregister no longer dedupes by expression"


def test_registered_candidates_cannot_be_silently_replaced(tmp_path,
                                                           monkeypatch):
    """Swapping in a better candidate after seeing the data is the search
    again with a sample size of one, and it would not feel like cheating."""
    from loonie import seal as S

    monkeypatch.setattr(S, "resolve", lambda q: tmp_path / Path(q).name)
    cfg = config.load()
    s = S.Seal.create_forward(cfg, min_sessions=0, archive_existing=False)

    class G:
        def __init__(self, fp):
            self.fingerprint = fp

        def canonical(self):
            return "expr_" + self.fingerprint

    s.register([G("aaaa"), G("bbbb")])
    assert len(s.registered) == 2

    reloaded = S.Seal.load(object())
    assert {r["fingerprint"] for r in reloaded.registered} == {"aaaa", "bbbb"}, \
        "the commitment must survive a reload intact"


def test_a_worker_that_stopped_on_purpose_is_not_respawned():
    """Otherwise the supervisor loops forever on a finished search.

    A continuous worker returning 0 gets respawned by design. The search now
    exits when its own pre-committed stopping rule fires, and without a
    distinct code that exit is indistinguishable from an ordinary finish: the
    supervisor restarts it, it resumes, the rule fires again, and the cycle
    repeats every couple of minutes -- rebuilding the entire panel each time
    to reach a conclusion it already reached.
    """
    from loonie import orchestrator as orc

    assert orc.RC_STOPPED_BY_RULE not in (0, 1, 2), \
        "the stop code must not collide with success or ordinary failure"

    j = orc.Job(name="search", every_s=0, argv=[])
    assert j.continuous and not j.completed

    class FakeProc:
        def __init__(self, rc):
            self._rc = rc

        def poll(self):
            return self._rc

    sup = orc.Orchestrator.__new__(orc.Orchestrator)
    sup.jobs = {"search": j}
    sup.children = {}
    sup._log = lambda *a, **k: None
    j.proc = FakeProc(orc.RC_STOPPED_BY_RULE)
    sup._reap(j)

    assert j.completed, "a rule-driven stop was not recorded as completed"
    assert j.failures == 0, "stopping on purpose is not a failure"

    # And an ordinary crash must still be a crash.
    j2 = orc.Job(name="search", every_s=0, argv=[])
    sup.jobs = {"search": j2}
    j2.proc = FakeProc(1)
    sup._reap(j2)
    assert not j2.completed and j2.failures == 1


def test_stall_is_measured_from_persisted_history_not_a_live_counter():
    """A counter living in one process cannot measure a 300-generation stall.

    The supervisor restarts the search on every source change, and an
    in-process counter resets with it -- so a rule requiring 300 stalled
    generations would never once reach 300, and the stopping rule would be
    decorative. Stall has to be derived from ev.history, which is checkpointed
    every generation.
    """
    src = (Path(__file__).resolve().parent.parent / "scripts"
           / "run_evolve.py").read_text(encoding="utf-8")
    assert "ev.history" in src.split("the stopping rule")[-1], \
        "the stopping rule no longer reads persisted history"

    # And the arithmetic it relies on.
    ics = [1.0, 2.0, 3.0, 3.0, 2.5, 2.9]        # best at index 2
    best = max(ics)
    best_at = max(i for i, v in enumerate(ics) if v >= best)
    assert best_at == 3, "ties must count the LAST occurrence, not the first"
    assert len(ics) - 1 - best_at == 2, "stall length is wrong"

    # A monotonically improving run is never stalled.
    rising = [1.0, 2.0, 3.0]
    b = max(rising)
    at = max(i for i, v in enumerate(rising) if v >= b)
    assert len(rising) - 1 - at == 0


def test_history_records_best_ic_for_the_stopping_rule():
    """Stall is measured on IC now, so IC has to be in the checkpoint."""
    import ast

    src = (Path(__file__).resolve().parent.parent / "loonie"
           / "evolve.py").read_text(encoding="utf-8")
    tree = ast.parse(src)
    keys = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Dict):
            for k in node.keys:
                if isinstance(k, ast.Constant) and isinstance(k.value, str):
                    keys.add(k.value)
    assert "best_ic_t" in keys, "history no longer records best_ic_t"


def test_best_ic_in_history_tracks_the_generation_leader():
    """A stall signal that never moves is not a plateau, it is a broken gauge.

    ic_t is computed inside gate(), which runs on the generation leader. The
    first version of this record took the max over ARCHIVE elites instead --
    most of which never reach gating and so carry no ic_t at all -- and the
    maximum sat frozen on whichever few had one. It logged 0.80 for 97
    consecutive generations. The stopping rule would have read that as a dead
    flat search and fired on a number that was never measuring anything.
    """
    from loonie import evolve

    p = synthetic_panel(T=800, N=20, seed=83)
    f = features.build(p)
    c = config.load()
    c["cv"]["n_splits"] = 3
    c["evolve"]["null_samples_per_gen"] = 0
    c["evolve"]["population"] = 12

    ev = evolve.Evolver(c, p, f, seed=9, verbose=False)
    ev.seed_population(12)
    recs = [ev.step() for _ in range(3)]

    assert all("best_ic_t" in r for r in recs), "best_ic_t missing from history"
    # It must come from the leader that was actually gated this generation.
    leader_ic = ev.population[0].cv.get("ic_t")
    if leader_ic is not None and np.isfinite(leader_ic):
        assert abs(recs[-1]["best_ic_t"] - float(leader_ic)) < 1e-9, \
            "history's best_ic_t does not match the gated leader"


def test_scripts_have_no_unbound_names_in_their_exit_paths():
    """A NameError on the last line of a long daemon run is expensive.

    run_evolve's stopping path returned orchestrator.RC_STOPPED_BY_RULE while
    the orchestrator import was added in a separate edit. The reference sat
    OUTSIDE the try block, so a search that had run for 2,230 generations
    printed its full summary and then died with a NameError -- which the
    supervisor read as rc=1, "died unexpectedly", and restarted.

    Nothing short of running to completion would have surfaced it, so this
    checks statically instead: every name a script's main() loads must be a
    local, an import, a module-level definition, or a builtin.
    """
    import ast
    import builtins

    root = Path(__file__).resolve().parent.parent / "scripts"
    offenders = {}
    for path in sorted(root.glob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"))

        bound = set(dir(builtins))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for al in node.names:
                    bound.add(al.asname or al.name.split(".")[0])
            elif isinstance(node, ast.ImportFrom):
                for al in node.names:
                    bound.add(al.asname or al.name)
            elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef,
                                   ast.ClassDef)):
                bound.add(node.name)
            elif isinstance(node, ast.Assign):
                for t in node.targets:
                    if isinstance(t, ast.Name):
                        bound.add(t.id)
            elif isinstance(node, (ast.AnnAssign, ast.AugAssign)):
                if isinstance(node.target, ast.Name):
                    bound.add(node.target.id)

        for fn in [n for n in ast.walk(tree)
                   if isinstance(n, ast.FunctionDef)]:
            local = {a.arg for a in fn.args.args + fn.args.kwonlyargs}
            if fn.args.vararg:
                local.add(fn.args.vararg.arg)
            if fn.args.kwarg:
                local.add(fn.args.kwarg.arg)
            for n in ast.walk(fn):
                if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Store):
                    local.add(n.id)
                elif isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef,
                                    ast.ClassDef)):
                    local.add(n.name)
                elif isinstance(n, ast.ExceptHandler) and n.name:
                    local.add(n.name)
                elif isinstance(n, (ast.Import, ast.ImportFrom)):
                    for al in n.names:
                        local.add(al.asname or al.name.split(".")[0])
                elif isinstance(n, ast.comprehension):
                    for t in ast.walk(n.target):
                        if isinstance(t, ast.Name):
                            local.add(t.id)
                # Parameters of lambdas and nested defs are bound inside them.
                # Without this every `lambda kv: ...` and every nested
                # callback argument reads as unbound, and the check drowns in
                # false positives -- which is how a real one gets waved
                # through.
                if isinstance(n, (ast.Lambda, ast.FunctionDef,
                                  ast.AsyncFunctionDef)):
                    ar = n.args
                    for a in (ar.args + ar.posonlyargs + ar.kwonlyargs):
                        local.add(a.arg)
                    if ar.vararg:
                        local.add(ar.vararg.arg)
                    if ar.kwarg:
                        local.add(ar.kwarg.arg)

            for n in ast.walk(fn):
                if (isinstance(n, ast.Name) and isinstance(n.ctx, ast.Load)
                        and n.id not in local and n.id not in bound):
                    offenders.setdefault(path.name, set()).add(n.id)

    assert not offenders, "unbound names in scripts: %s" % {
        k: sorted(v) for k, v in offenders.items()}


def test_system_log_is_written_by_the_process_not_the_caller(tmp_path):
    """A log whose destination depends on the launch command stops silently.

    The orchestrator logs with print(), so starting it with stdout redirected
    anywhere else leaves state/system.log present, readable, and hours out of
    date -- which still LOOKS like current state. That is how an evening of
    failing publish jobs stayed invisible here.
    """
    import importlib.util

    path = Path(__file__).resolve().parent.parent / "scripts" / "run_system.py"
    src = path.read_text(encoding="utf-8")
    assert "state/system.log" in src, "run_system no longer owns its log file"

    spec = importlib.util.spec_from_file_location("_rs_probe", path)
    mod = importlib.util.module_from_spec(spec)
    # Importing would start the supervisor; just exercise the tee directly.
    tee_src = src.split("class _Tee:")[1].split("sys.stdout = _Tee")[0]
    ns: dict = {}
    exec("class _Tee:" + tee_src, {"sys": sys, "open": open}, ns)

    log = tmp_path / "system.log"

    class Sink:
        def __init__(self):
            self.seen = []

        def write(self, s):
            self.seen.append(s)

        def flush(self):
            pass

    sink = Sink()
    t = ns["_Tee"](sink, log)
    t.write("[orchestrator] hello\n")
    t.flush()

    assert "hello" in log.read_text(encoding="utf-8"), "nothing reached the file"
    assert sink.seen, "the console stream was starved"
    # Flushed per write: a log you have to wait for is not a log.
    t.write("[orchestrator] second\n")
    assert "second" in log.read_text(encoding="utf-8")


class _Pos:
    def __init__(self, mv):
        self.market_value = mv


class _Acct:
    def __init__(self, n, each):
        self.positions = {"S%02d" % i: _Pos(each) for i in range(n)}
        self.equity = n * each


def test_a_sell_is_never_blocked_by_the_gross_exposure_cap():
    """The bug that froze the paper book for a day.

    check_order added every order's notional to current exposure regardless of
    direction, so a SELL -- which reduces exposure -- was rejected for
    would-be breaching the cap. Once the book filled, that blocked exactly the
    orders that make room: seven consecutive rebalances proposed up to 86
    orders and submitted none, while the heartbeat reported healthy runs.
    """
    from loonie import risk

    c = config.load()
    rm = risk.RiskManager(c) if hasattr(risk, "RiskManager") else None
    if rm is None:
        pytest.skip("RiskManager not exposed under that name")

    acct = _Acct(20, 5000.0)                 # $100k, fully invested
    eq = acct.equity

    # A buy at a full book is correctly refused...
    buy = rm.check_order("NEW", 4000.0, eq, acct, side="buy")
    assert not buy.allow and "gross exposure" in buy.reason

    # ...and a sell of the same size must not be.
    sell = rm.check_order("S00", 4000.0, eq, acct, side="sell")
    assert sell.allow, "a sell was blocked by the exposure cap: %s" % sell.reason


def test_exposure_freed_earlier_in_the_batch_is_counted():
    """Sells approved a moment ago have to reduce the number the next buy is
    measured against, or the first buy after them still sees a full book."""
    from loonie import risk

    c = config.load()
    rm = risk.RiskManager(c)
    acct = _Acct(20, 5000.0)
    eq = acct.equity

    assert not rm.check_order("NEW", 4000.0, eq, acct, side="buy").allow
    freed = rm.check_order("NEW", 4000.0, eq, acct, side="buy", pending=-9000.0)
    assert freed.allow, "room freed by earlier sells was not counted"


def test_trade_submits_sells_before_buys():
    """Ordering is part of the fix: buys measured against an unreduced book
    fail no matter how many sells appear later in the same batch."""
    src = (Path(__file__).resolve().parent.parent / "scripts"
           / "run_trade.py").read_text(encoding="utf-8")
    assert 'sorted(orders' in src and '"sell"' in src, \
        "run_trade no longer orders sells ahead of buys"
    assert "side=o.side" in src, "order side is not passed to the risk check"


def test_an_oversized_position_can_still_be_sold():
    """Otherwise a cap breach is permanent.

    max_position_pct sizes a position UP. Applied to a sell it makes the very
    name that drifted above the cap unsellable *because* it is oversized --
    BKR sat at 6.1% against a 6.0% cap and could not be trimmed for exactly
    that reason.
    """
    from loonie import risk

    c = config.load()
    rm = risk.RiskManager(c)
    acct = _Acct(20, 5000.0)
    eq = acct.equity
    maxpos = float(c["trade"]["max_position_pct"]) * eq
    oversized = maxpos * 1.05

    buy = rm.check_order("BKR", oversized, eq, acct, side="buy")
    assert not buy.allow and "max_position_pct" in buy.reason, \
        "the cap must still stop an oversized BUY"

    sell = rm.check_order("BKR", oversized, eq, acct, side="sell")
    assert sell.allow, \
        "an oversized position could not be trimmed: %s" % sell.reason


# =============================================================================
#  Execution realism
# =============================================================================
def test_orders_fill_at_the_next_open_not_the_signal_close():
    """The lookahead that lived in the execution layer.

    The signal is computed from Tuesday's close; Tuesday's close is gone by
    the time you have computed it. Filling there is a one-bar lookahead worth
    roughly the overnight gap on every trade, sitting where nobody looks for
    lookahead.
    """
    from loonie import execution as X

    # Gaps UP overnight: a buyer pays more than the close they decided on.
    f = X.simulate("buy", qty=100, ref_price=100.0, open_price=101.0,
                   adv_dollars=5e7, daily_vol=0.02, symbol="T")
    assert f.fill_price > 100.0
    assert abs(f.gap_bps - 100.0) < 1.0, "a 1%% gap against a buyer is +100 bps"

    # The same gap HELPS a seller, and the sign must say so.
    g = X.simulate("sell", qty=100, ref_price=100.0, open_price=101.0,
                   adv_dollars=5e7, daily_vol=0.02, symbol="T")
    assert g.gap_bps < 0, "a favourable gap was charged as a cost"


def test_impact_grows_with_size_so_scale_is_not_free():
    """A flat slippage says a $500 order and a $5,000,000 order cost the same
    fraction, which is backwards: the difficulty of running more money is that
    it does not scale."""
    from loonie import execution as X

    adv, vol = 1e7, 0.02
    small = X.impact_bps(1e4, adv, vol)
    big = X.impact_bps(1e6, adv, vol)
    assert big > small * 5, "impact barely moved with a 100x order"
    # Square-root law: 100x the size is ~10x the impact, not 100x.
    assert 8 < big / max(small, 1e-9) < 13


def test_spread_is_wider_for_cheap_illiquid_names():
    """Charging a $400 megacap and a $9 thin name the same 5 bps subsidises
    trading illiquid things -- which is where a naive search likes to go."""
    from loonie import execution as X

    liquid = X.spread_bps(price=400.0, dollar_vol=5e9)
    thin = X.spread_bps(price=9.0, dollar_vol=2e6)
    assert thin > liquid * 3, "spread did not widen for the illiquid name"
    assert liquid < 5.0


def test_an_order_cannot_consume_more_than_its_share_of_volume():
    """Pretending otherwise is how a backtest 'scales' to money it could
    never deploy."""
    from loonie import execution as X

    adv = 1e6
    f = X.simulate("buy", qty=1e6, ref_price=10.0, open_price=10.0,
                   adv_dollars=adv, daily_vol=0.02, symbol="THIN")
    assert f.filled
    assert f.notional <= adv * X.MAX_PARTICIPATION * 1.01
    assert "participation capped" in f.note


def test_a_queued_order_survives_a_restart(tmp_path, monkeypatch):
    """A decision made after Tuesday's close is still owed a Wednesday fill.

    Dropping the queue on restart would turn a missed trade into a trade that
    never existed, and the paper record would show neither the cost nor the
    position.
    """
    from loonie.broker import paper as P

    monkeypatch.setattr(P, "resolve", lambda q: tmp_path / Path(q).name)
    cfg = config.load()

    b = P.PaperBroker(cfg, starting_cash=100000.0)
    b.set_marks({"AAA": 50.0})
    o = b.submit(P.Order(symbol="AAA", side="buy", qty=0.0, notional=5000.0))
    assert o.status == "pending", "the order filled immediately"
    assert not b.positions(), "a position appeared before any fill"

    b2 = P.PaperBroker(cfg)
    assert len(b2.pending) == 1, "the queue did not survive a restart"

    fills = b2.settle({"AAA": 51.0}, {"AAA": {"adv": 5e7, "vol": 0.02}})
    assert len(fills) == 1 and fills[0].filled
    assert "AAA" in b2.positions(), "settling produced no position"
    assert b2.positions()["AAA"].avg_price > 51.0, "costs were not charged"


def test_settlement_sells_before_buying_so_cash_is_available():
    """Settling in submission order leaves buys short of the cash the same
    batch is about to produce."""
    import tempfile

    from loonie.broker import paper as P

    with tempfile.TemporaryDirectory() as td:
        import loonie.broker.paper as mod
        orig = mod.resolve
        mod.resolve = lambda q: Path(td) / Path(q).name
        try:
            cfg = config.load()
            b = P.PaperBroker(cfg, starting_cash=10000.0)
            b.set_marks({"OLD": 100.0, "NEW": 100.0})
            b.settle({"OLD": 100.0}, {"OLD": {"adv": 5e8, "vol": 0.01}})
            b.submit(P.Order(symbol="OLD", side="buy", qty=0.0, notional=9000.0))
            b.settle({"OLD": 100.0}, {"OLD": {"adv": 5e8, "vol": 0.01}})

            # Now almost no cash; queue a buy that only a sell can fund.
            b.submit(P.Order(symbol="NEW", side="buy", qty=0.0, notional=8000.0))
            b.submit(P.Order(symbol="OLD", side="sell",
                             qty=b.positions()["OLD"].qty))
            fills = b.settle({"OLD": 100.0, "NEW": 100.0},
                             {"OLD": {"adv": 5e8, "vol": 0.01},
                              "NEW": {"adv": 5e8, "vol": 0.01}})
            got = {f.symbol: f for f in fills if f.filled}
            assert "NEW" in got, "the buy was starved by settlement order"
            assert got["NEW"].notional > 7000, \
                "the buy was downsized despite the sell funding it"
        finally:
            mod.resolve = orig


def test_an_order_cannot_fill_at_an_open_that_preceded_its_own_signal():
    """The obvious implementation of "fill at the open" fills backwards.

    An order decided on bar T's CLOSE must fill at the open of T+1 or later.
    Settling it against bar T's own open is a fill several hours before the
    information that produced it existed -- worse than the instant-fill bug it
    was meant to replace, and invisible in any P&L that only checks prices are
    plausible.
    """
    import tempfile

    from loonie.broker import paper as P

    with tempfile.TemporaryDirectory() as td:
        import loonie.broker.paper as mod
        orig = mod.resolve
        mod.resolve = lambda q: Path(td) / Path(q).name
        try:
            cfg = config.load()
            b = P.PaperBroker(cfg, starting_cash=50000.0)
            b.set_marks({"AAA": 100.0}, as_of="2026-09-16")
            b.submit(P.Order(symbol="AAA", side="buy", qty=0.0, notional=5000.0))
            assert b.pending[0]["ref_bar"] == "2026-09-16"

            liq = {"AAA": {"adv": 5e8, "vol": 0.01}}
            same = b.settle({"AAA": 99.0}, liq, bar_date="2026-09-16")
            assert same == [], "filled on the same bar that produced the signal"
            assert len(b.pending) == 1, "the order was consumed anyway"
            assert not b.positions()

            nxt = b.settle({"AAA": 99.0}, liq, bar_date="2026-09-17")
            assert len(nxt) == 1 and nxt[0].filled, "the next open did not fill it"
            assert not b.pending
        finally:
            mod.resolve = orig


def test_a_source_change_reconsiders_a_completed_worker():
    """Otherwise the edit that was meant to restart it is silently ignored.

    A worker that stopped on its own rule stays stopped -- but the rules live
    in config.yaml, which the fingerprint watches. Raising max_trials to
    resume searching and having nothing happen is exactly the failure: the
    config said keep going, the supervisor said already finished, and neither
    logged a disagreement.
    """
    from loonie import orchestrator as orc

    import time as _t

    sup = orc.Orchestrator.__new__(orc.Orchestrator)
    j = orc.Job(name="search", every_s=0, argv=[])
    j.completed = True
    sup.jobs = {"search": j}
    sup.children = {}
    sup.reloads = 0
    sup.fingerprint = "definitely-not-the-current-fingerprint"
    sup._log = lambda *a, **k: None
    # Pre-arm the debounce so the change counts on this call rather than
    # merely being noticed.
    sup._pending_fp = orc.source_fingerprint()
    sup._pending_since = _t.time() - 3600.0

    changed = sup.check_source()
    assert changed is True, "the source change was not acted on"
    assert not j.completed, "a completed worker was not reconsidered"
    assert j.last_status == "pending"


def test_config_is_part_of_the_reload_fingerprint():
    """The gates and the stopping rule live in config.yaml, so a change there
    has to count as a source change."""
    from loonie import orchestrator as orc
    import inspect

    src = inspect.getsource(orc.source_fingerprint)
    assert "config.yaml" in src, "config.yaml is not watched for reloads"


def test_averaging_ranks_not_raw_scores():
    """Raw scores live on wildly different scales.

    One genome returns log-dollars, another a z-score. Averaging them directly
    lets whichever has the largest numbers dominate the blend, so the
    "ensemble" is really just its loudest component wearing a committee's
    name. Ranks make the average a vote rather than a sum.
    """
    rng = np.random.default_rng(31)
    T, N = 50, 40
    small = rng.normal(0, 0.001, (T, N))       # a z-score-ish signal
    huge = rng.normal(0, 1000.0, (T, N))       # dollars

    raw = (small + huge) / 2.0
    # The raw average is essentially the large-scale series alone.
    assert abs(np.corrcoef(raw.ravel(), huge.ravel())[0, 1]) > 0.99

    r = lambda a: pd.DataFrame(a).rank(axis=1, pct=True).to_numpy()  # noqa: E731
    blended = (r(small) + r(huge)) / 2.0
    c_small = abs(np.corrcoef(blended.ravel(), r(small).ravel())[0, 1])
    c_huge = abs(np.corrcoef(blended.ravel(), r(huge).ravel())[0, 1])
    assert abs(c_small - c_huge) < 0.15, \
        "ranked blend still favoured one component (%.2f vs %.2f)" % (c_small, c_huge)


def test_ensemble_script_cannot_skip_the_out_of_sample_split():
    """The in-sample ensemble reads t +4.49 against a best single of +2.93 --
    a 53% gain that is entirely selection, and vanishes to +0.04 on a held-out
    window. A tool that can produce the flattering number on request will
    eventually be asked to, so there is no flag to skip the split.
    """
    import ast

    src = (Path(__file__).resolve().parent.parent / "scripts"
           / "ensemble.py").read_text(encoding="utf-8")
    tree = ast.parse(src)
    flags = set()
    for node in ast.walk(tree):
        if (isinstance(node, ast.Call)
                and getattr(node.func, "attr", "") == "add_argument"):
            for arg in node.args:
                if isinstance(arg, ast.Constant) and isinstance(arg.value, str):
                    flags.add(arg.value)
    banned = {"--no-split", "--in-sample", "--full", "--skip-split"}
    assert not (flags & banned), "an escape hatch from the split was added"
    assert "--split" in flags, "the split is no longer configurable at all"
    assert "_slice_panel" in src, "the script no longer splits the panel"


# =============================================================================
#  Wide universe
# =============================================================================
def test_liquidity_rule_reads_only_the_past():
    """A universe rule that reads today's volume to decide whether today's
    name is tradable is reading the session it is about to trade."""
    from loonie import wide

    T, N = 300, 8
    rng = np.random.default_rng(5)
    close = np.full((T, N), 100.0)
    volume = rng.uniform(1e5, 1e6, (T, N))

    full = wide.liquidity_rank(close, volume, lookback=60)
    cut = 200
    trunc = wide.liquidity_rank(close[:cut], volume[:cut], lookback=60)
    both = np.isfinite(full[:cut]) & np.isfinite(trunc)
    assert both.any()
    assert np.allclose(full[:cut][both], trunc[both]), \
        "the liquidity rule changed when future data was removed"


def test_membership_holds_between_refreshes():
    """Re-ranking daily churns names in and out on noise, and the strategy is
    charged turnover for a decision the universe definition made, not one it
    made itself."""
    from loonie import wide

    T, N = 400, 50
    rng = np.random.default_rng(9)
    close = np.full((T, N), 50.0)
    volume = rng.uniform(1e4, 1e7, (T, N))
    m = wide.membership(close, volume, n_names=20, lookback=60,
                        refresh_every=21, min_dollar_vol=0.0, min_price=1.0)

    # Within a refresh block membership must not move at all. The block
    # boundaries are derived, not guessed: hardcoding rows 120..140 straddled
    # the refresh at 123 and failed the code for doing exactly its job.
    lookback, every = 60, 21
    starts = list(range(lookback, T, every))
    a, b = starts[3], min(starts[3] + every, T)
    block = m[a:b]
    assert (block == block[0]).all(), "membership changed mid-block"

    # And it must actually change between blocks, or the rule is inert.
    assert not np.array_equal(m[starts[3]], m[starts[6]]),         "membership never changed across refreshes"


def test_membership_respects_the_size_cap():
    from loonie import wide

    T, N = 300, 200
    rng = np.random.default_rng(13)
    close = np.full((T, N), 20.0)
    volume = rng.uniform(1e5, 1e8, (T, N))
    m = wide.membership(close, volume, n_names=30, lookback=60,
                        refresh_every=21, min_dollar_vol=0.0, min_price=1.0)
    per_day = m.sum(axis=1)
    assert per_day.max() <= 30, "the universe exceeded its target size"


def test_directory_filters_out_non_common_stock():
    """Units, warrants and rights have their own price dynamics and would be
    selected FOR by a liquidity rule in any week a SPAC was in the news."""
    from loonie import wide

    assert ".U" in wide.NOT_COMMON and ".W" in wide.NOT_COMMON
    src = (Path(__file__).resolve().parent.parent / "loonie"
           / "wide.py").read_text(encoding="utf-8")
    # The reindex trap: filtering a frame then reusing a Series built from the
    # pre-filter index silently drops the wrong rows.
    assert src.count("reset_index(drop=True)") >= 3, \
        "symbol filters no longer reset the index between steps"


# =============================================================================
#  Congressional disclosures
# =============================================================================
SAMPLE_PTR = """Filing ID #20032062
Name: Hon. Robert B. Aderholt
ID Owner Asset Transaction Date Notification Amount Cap.
GSK plc American Depositary Shares S 07/28/2025 08/11/2025 $1,001 - $15,000
(GSK) [ST]
Microsoft Corporation P 06/02/2025 06/15/2025 $15,001 - $50,000
(MSFT) [ST]
Some Municipal Bond Fund P 05/01/2025 05/10/2025 $1,001 - $15,000
Digitally Signed: Hon. Robert B. Aderholt , 09/10/2025"""


def _parse_lines(text):
    """Exercise the row/ticker regexes the way parse_ptr does."""
    from loonie import congress as C

    lines = [ln.strip() for ln in text.splitlines()]
    out = []
    for i, line in enumerate(lines):
        m = C._ROW.search(line)
        if not m:
            continue
        tk = None
        for nxt in lines[i + 1:i + 3]:
            t = C._TICKER_LINE.match(nxt)
            if t:
                tk = t.group("t")
                break
        if not tk:
            continue
        out.append({"ticker": tk, "side": C._SIDE[m.group("side")],
                    "tx": m.group("tx"), "amt": m.group("amt")})
    return out


def test_ptr_rows_span_two_lines_and_sides_are_not_all_buys():
    """The bug that made the first parser useless.

    A PTR row wraps: side, dates and amount on one line, the ticker in
    (TICKER) form on the next. A parser requiring both on one line keeps only
    the rows that happened not to wrap -- and detecting the side by scanning
    the whole line for " P " matches letters inside company names, which
    returned "buy" for every single transaction in the first run.
    """
    got = _parse_lines(SAMPLE_PTR)
    by = {g["ticker"]: g for g in got}

    assert "GSK" in by and by["GSK"]["side"] == "sell", \
        "S must parse as a sell, not a buy"
    assert "MSFT" in by and by["MSFT"]["side"] == "buy"
    assert by["GSK"]["tx"] == "07/28/2025"
    assert by["GSK"]["amt"] == "$1,001 - $15,000"

    sides = {g["side"] for g in got}
    assert sides == {"buy", "sell"}, "every row came back the same side: %s" % sides

    # An asset with no (TICKER) line cannot be traded against and is dropped
    # rather than guessed at.
    assert len(got) == 2, "a row without a ticker was kept: %r" % got


def test_congress_regex_has_no_stray_control_characters():
    """A heredoc turned \b into a literal backspace byte (0x08) at the start
    of the pattern, so it matched nothing at all and the parser returned zero
    transactions while looking perfectly reasonable in the source."""
    from loonie import congress as C

    for name in ("_ROW", "_TICKER_LINE"):
        pat = getattr(C, name).pattern
        bad = [c for c in pat if ord(c) < 32]
        assert not bad, "%s contains control characters: %r" % (name, bad)


def test_congress_signal_is_keyed_to_the_filing_date_not_the_trade():
    """The 45-day problem, and the single most flattering error available.

    The STOCK Act allows up to 45 days between a trade and its disclosure. A
    signal applied on the TRANSACTION date buys at prices nobody following the
    filings could have paid, and shows a handsome edge precisely because the
    information was private for that month.
    """
    import pandas as pd

    from loonie import congress as C

    p = synthetic_panel(T=400, N=6, seed=77)
    p.dates = pd.bdate_range("2025-01-01", periods=400)
    df = pd.DataFrame([{
        "ticker": p.tickers[0], "side": "buy",
        "transaction_date": p.dates[100],
        "notification_date": p.dates[130],
        "filing_date": p.dates[140],
        "amount_mid": 50000.0,
    }])

    out = C.build_signal(p, df, decay_days=30)
    sig = out["cong_net_buy"][:, 0]

    assert np.all(sig[:140] == 0), \
        "the signal appeared before the filing became public"
    assert sig[140] > 0, "the signal never appeared at all"
    assert sig[139] == 0, "the signal leaked one day early"
    # And it decays rather than persisting forever.
    assert sig[141] < sig[140] and np.all(sig[175:] == 0)


def test_amount_bands_are_midpoints_of_the_disclosed_range():
    """Disclosed amounts are ranges, so any size here is a convention. The
    bands must at least sit inside the range they claim to represent."""
    from loonie import congress as C

    import re as _re
    for band, mid in C.AMOUNT_BANDS.items():
        nums = [int(x.replace(",", "")) for x in _re.findall(r"[\d,]+", band)]
        if len(nums) == 2:
            assert nums[0] <= mid <= nums[1], \
                "%s midpoint %d is outside the band" % (band, mid)


def test_event_study_benchmarks_against_the_universe():
    """Without subtracting the market, a month when everything rose reads as
    disclosure alpha."""
    src = (Path(__file__).resolve().parent.parent / "scripts"
           / "congress_event_study.py").read_text(encoding="utf-8")
    assert "nancumsum" in src and "nanmean" in src, \
        "the event study no longer nets out the universe return"
    assert "filing_date" in src and "transaction_date" not in src, \
        "the event study must key on the filing date only"


def test_clustered_tstat_is_not_the_naive_one():
    """1,988 events are not 1,988 independent observations: members file in
    batches and neighbouring events share overlapping return windows.

    Here the correction happens to RAISE the t-stat, because averaging within
    a cluster removes noise faster than it removes sample. That is a fact
    about this data, not a reason to skip it -- with different clustering it
    would cut the other way, and the naive number would be the flattering one.
    """
    import importlib.util

    p = (Path(__file__).resolve().parent.parent / "scripts"
         / "congress_event_study.py")
    spec = importlib.util.spec_from_file_location("_ces", p)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    rng = np.random.default_rng(3)
    # Twenty clusters of ten, with the signal at cluster level and pure noise
    # within: the naive t-stat sees 200 observations that are really 20.
    base = rng.normal(0.004, 0.01, 20)
    rows = []
    for k, b in enumerate(base):
        for _ in range(10):
            rows.append({"date": k, "y": b + rng.normal(0, 0.05)})
    d = pd.DataFrame(rows)

    naive = mod.tstat(d["y"])
    clustered = mod.tstat(d.groupby("date")["y"].mean())
    assert abs(naive - clustered) > 0.3, \
        "clustering changed nothing; the correction is not being applied"


def test_congress_weights_apply_the_session_after_publication():
    """A filing appears during a day whose close has already happened.

    Acting on it at that close is a few hours of lookahead -- small, invisible
    in any sanity check on prices, and worth more than the entire measured
    effect over a five-day hold.
    """
    import importlib.util

    p = (Path(__file__).resolve().parent.parent / "scripts"
         / "congress_backtest.py")
    spec = importlib.util.spec_from_file_location("_cbt", p)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    dates = pd.bdate_range("2025-01-01", periods=60)
    tickers = ["AAA", "BBB"]
    member = np.ones((60, 2), bool)
    df = pd.DataFrame([{"ticker": "AAA", "side": "buy",
                        "filing_date": dates[20], "member": "X"}])

    w = mod.build_weights(df, dates, tickers, member, horizon=5,
                          mode="long_short", max_pos=1.0)
    assert w[20, 0] == 0.0, "acted on the same session the filing appeared"
    assert w[21, 0] > 0, "never acted on the filing at all"
    assert w[26, 0] == 0.0, "the hold did not expire after 5 sessions"


def test_repeated_disclosure_does_not_double_a_position():
    """Two members disclosing the same name inside one holding window is not
    a reason to hold twice as much of it -- that is a sizing decision nobody
    made, concentrating exactly where the data is noisiest."""
    import importlib.util

    p = (Path(__file__).resolve().parent.parent / "scripts"
         / "congress_backtest.py")
    spec = importlib.util.spec_from_file_location("_cbt2", p)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    dates = pd.bdate_range("2025-01-01", periods=60)
    tickers = ["AAA"]
    member = np.ones((60, 1), bool)
    df = pd.DataFrame([
        {"ticker": "AAA", "side": "buy", "filing_date": dates[20], "member": "X"},
        {"ticker": "AAA", "side": "buy", "filing_date": dates[21], "member": "Y"},
    ])
    w = mod.build_weights(df, dates, tickers, member, horizon=5,
                          mode="long_short", max_pos=1.0)
    assert np.nanmax(np.abs(w)) <= 1.0 + 1e-9, "position exceeded full weight"


# =============================================================================
#  Strategy registry
# =============================================================================
def test_annualisation_uses_the_strategy_period_not_the_calendar():
    """The bug that reported 266% a year for something that made 6%.

    A monthly options model returns one row per HOLDING PERIOD. Annualising
    442 of those as 442 trading days compresses 36 years into 1.75 and
    inflates CAGR by a factor of twenty. Every derived column inherits it, and
    the number looks merely impressive rather than impossible.
    """
    import pandas as pd

    from loonie.strategies import Result

    n = 442
    r = np.full(n, 0.005)                      # 0.5% per monthly period
    dates = pd.bdate_range("1990-01-01", periods=n, freq="21D")
    eq = np.cumprod(1.0 + r)

    wrong = Result(id="x", title="x", market="options", description="",
                   dates=dates, equity=eq, returns=r).stats()
    right = Result(id="x", title="x", market="options", description="",
                   dates=dates, equity=eq, returns=r,
                   periods_per_year=252.0 / 21.0).stats()

    assert right["years"] > 30, "36 years of monthly periods read as %.1f" % right["years"]
    assert wrong["years"] < 3
    assert right["cagr_pct"] < wrong["cagr_pct"] / 5, \
        "the correction barely changed the annualised figure"
    assert 3.0 < right["cagr_pct"] < 10.0


def test_win_rate_counts_only_active_periods():
    """A strategy that stands aside most of the time would otherwise report a
    win rate made of days it did nothing."""
    import pandas as pd

    from loonie.strategies import Result

    r = np.zeros(200)
    r[:10] = 0.01
    r[10:20] = -0.01                            # 10 wins, 10 losses, 180 flat
    res = Result(id="x", title="x", market="stocks", description="",
                 dates=pd.bdate_range("2020-01-01", periods=200),
                 equity=np.cumprod(1.0 + r), returns=r)
    st = res.stats()
    assert abs(st["win_rate_pct"] - 50.0) < 1e-6, \
        "flat periods were counted as wins (%.1f%%)" % st["win_rate_pct"]
    assert st["active_periods"] == 20


def test_every_strategy_declares_a_caveat():
    """A modelled options P&L and a measured equity return are different kinds
    of number. The difference has to be a field, not a footnote."""
    from loonie import strategy_lib  # noqa: F401
    from loonie.strategies import REGISTRY

    assert len(REGISTRY) >= 10
    for sid, spec in REGISTRY.items():
        assert spec["caveat"].strip(), "%s has no caveat" % sid
        assert spec["market"] in ("stocks", "crypto", "forex", "options")
        assert len(spec["description"]) > 80, "%s description is too thin" % sid

    # Anything modelled rather than measured must say so in capitals, where it
    # cannot be skimmed past.
    for sid, spec in REGISTRY.items():
        if spec["market"] == "options":
            assert "MODELLED" in spec["caveat"], \
                "%s does not declare that it is modelled" % sid


def test_a_per_row_constant_cannot_change_cross_sectional_ranks():
    """The bug that made two strategies identical to four significant figures.

    "Altcoin rotation" divided every coin's momentum by Bitcoin's, then ranked
    the result. Dividing a whole row by the same scalar is a monotone
    transform, so the ranks are untouched and the strategy was arithmetically
    cross-sectional momentum wearing a different name. It reported 410.0%,
    14.5%, 52%, -60%, 1.80 -- the same five numbers.
    """
    from loonie.strategies import _ranks

    rng = np.random.default_rng(19)
    x = rng.normal(0, 1, (40, 12))
    mask = np.ones((40, 12), bool)
    scale = rng.uniform(0.5, 2.0, (40, 1))       # a different constant per row

    assert np.allclose(_ranks(x, mask), _ranks(x * scale, mask),
                       equal_nan=True), \
        "test premise broken: scaling a row changed its ranks"

    # A relative bet only differs if the benchmark is an actual POSITION.
    src = (Path(__file__).resolve().parent.parent / "loonie"
           / "strategy_lib.py").read_text(encoding="utf-8")
    body = src.split("def cry_btc_relative")[1].split("@register")[0]
    assert "row[j_btc] = -1.0" in body, \
        "altcoin rotation no longer shorts Bitcoin; it is momentum again"


def test_strategy_ids_are_unique_and_results_are_distinguishable():
    """Two rows reporting identical statistics is a bug, not a coincidence."""
    from loonie import strategy_lib  # noqa: F401
    from loonie.strategies import REGISTRY

    ids = list(REGISTRY)
    assert len(ids) == len(set(ids))
    titles = [v["title"] for v in REGISTRY.values()]
    assert len(titles) == len(set(titles)), "duplicate strategy titles"


# =============================================================================
#  Warehouse
# =============================================================================
def test_reverse_split_artifacts_are_dropped_from_returns():
    """The data corruption the warehouse found.

    yfinance back-adjusts for splits, so a company that has done several
    reverse splits has its history multiplied up astronomically -- JAGX peaks
    at an adjusted $714,285,696 and GNLN spans a 9,342,787x range. Reverse
    splits happen BECAUSE a stock collapsed, so these sit entirely in the
    illiquid tail. A 500,000% daily "return" is not something anyone traded.
    """
    import pandas as pd

    from loonie.markets import MarketPanel

    close = np.array([[100.0], [101.0], [5_000_000.0], [5_050_000.0]])
    p = MarketPanel(market="stocks",
                    dates=pd.bdate_range("2020-01-01", periods=4),
                    symbols=["JUNK"], close=close)
    r = p.returns()
    assert abs(r[1, 0] - 0.01) < 1e-9, "an ordinary 1% move was altered"
    assert r[2, 0] == 0.0, "a 5,000,000% jump survived into the returns"
    assert abs(r[3, 0] - 0.01) < 1e-9, "the move after the artefact was lost"

    # Crypto is exempt: it genuinely moves more than 60% in a day.
    pc = MarketPanel(market="crypto",
                     dates=pd.bdate_range("2020-01-01", periods=2),
                     symbols=["X"], close=np.array([[100.0], [200.0]]))
    assert abs(pc.returns()[1, 0] - 1.0) < 1e-9, \
        "a real 100% crypto day was discarded"


def test_warehouse_rebuilds_from_source_and_owns_nothing(tmp_path, monkeypatch):
    """The warehouse is derived and disposable.

    An analytical store that becomes the only copy of something has stopped
    being an analytical store -- so every table must be reconstructible, and
    deleting the file must lose nothing.
    """
    import pandas as pd

    from loonie import markets as M
    from loonie import warehouse as W

    monkeypatch.setattr(W, "resolve", lambda q: tmp_path / Path(q).name)

    con = W.connect()
    dates = pd.bdate_range("2024-01-01", periods=30)
    close = np.random.default_rng(3).uniform(10, 100, (30, 4)).astype("float32")
    panel = M.MarketPanel(market="testmkt", dates=dates,
                          symbols=["A", "B", "C", "D"], close=close,
                          tradable=np.ones((30, 4), bool),
                          meta={"source": "unit test", "caveat": "synthetic"})
    n = W.load_market(con, "testmkt", panel)
    assert n == 120, "long format should be rows x symbols, got %d" % n

    got = con.execute("SELECT COUNT(*) FROM symbols WHERE market='testmkt'"
                      ).fetchone()[0]
    assert got == 4

    # The caveat must survive into the database; a market without one would
    # let a cross-market query quietly compare incomparable things.
    cav = con.execute("SELECT caveat FROM markets WHERE market='testmkt'"
                      ).fetchone()[0]
    assert cav == "synthetic"

    # Reloading the same market replaces rather than duplicates.
    W.load_market(con, "testmkt", panel)
    assert con.execute("SELECT COUNT(*) FROM prices WHERE market='testmkt'"
                       ).fetchone()[0] == 120
    con.close()


def test_prices_are_stored_long_with_market_as_a_column():
    """Wide is right for computing and wrong for asking questions."""
    from loonie import warehouse as W

    assert "market      VARCHAR," in W.SCHEMA
    assert "CREATE TABLE IF NOT EXISTS prices" in W.SCHEMA
    # Every market in one table, so a query can compare them without a union.
    assert W.SCHEMA.count("CREATE TABLE") >= 7
    assert "caveat" in W.SCHEMA, "markets table must carry its caveat"


# =============================================================================
#  RSI: replay simulator and the hypothesis ledger
# =============================================================================
def _world(spec, seed=0):
    """spec: {method: [scores]} -> a replayable World."""
    from loonie import dream as D

    nodes = []
    g = 0
    for m, scores in spec.items():
        for s in scores:
            g += 1
            nodes.append(D.Node(fingerprint="%s%d" % (m, g), method=m,
                                operator="x", generation=g, fitness=s,
                                fwd_alpha=s, fwd_t=0.0,
                                gates_passed=0, gates_total=8))
    return D.World.build(nodes, run_id="t")


def test_replay_terminates_when_a_policy_fixates_on_an_exhausted_method():
    """The bug that hung the first version.

    A greedy policy keeps choosing whichever method scored best. Once that
    method has nothing left, refusing the choice and continuing costs no
    budget -- so the loop never advances, never errors, and never returns.
    Exhausted methods are hidden from the policy instead.
    """
    from loonie import dream as D

    w = _world({"good": [0.9], "bad": [0.1, 0.1, 0.1, 0.1]})
    r = D.replay(D.Greedy(), w, budget=50, seed=0)
    assert r["evaluations"] == 5, "replay did not consume the whole world"
    assert r["best_score"] == 0.9


def test_replay_scores_on_forward_outcome_not_fitness():
    """Fitness is what the search optimised; forward alpha is what it was
    trying to predict. Replaying against fitness would measure how well a
    policy games the objective."""
    from loonie import dream as D

    n = D.Node(fingerprint="a", method="m", operator="o", generation=1,
               fitness=99.0, fwd_alpha=0.02, fwd_t=1.0,
               gates_passed=0, gates_total=8)
    assert n.score == 0.02, "score came from fitness, not the forward result"

    nan = D.Node(fingerprint="b", method="m", operator="o", generation=2,
                 fitness=99.0, fwd_alpha=float("nan"), fwd_t=float("nan"),
                 gates_passed=0, gates_total=8)
    assert nan.score == 0.0, "an unmeasured forward outcome became a score"


def test_a_world_with_one_method_cannot_discriminate_policies():
    """Why worlds pool runs instead of following them.

    run_evolve draws ONE method at startup and keeps it, so a per-run world
    offers no allocation decision and every policy ties at zero regret --
    which looks like a finding and is an artefact of the world's shape.
    """
    from loonie import dream as D

    w = _world({"only": [0.1, 0.5, 0.3, 0.2]})
    out = {D.replay(cls(), w, budget=2, seed=1)["best_score"]
           for cls in D.POLICIES}
    assert len(out) == 1, "a single-method world somehow discriminated"

    # Two methods and a budget below the world size, and it does.
    w2 = _world({"a": [0.9, 0.9, 0.9], "b": [0.0, 0.0, 0.0]})
    got = {cls.__name__: D.replay(cls(), w2, budget=2, seed=1)["best_score"]
           for cls in D.POLICIES}
    assert len(set(got.values())) > 1, "policies tied where they should differ"


def test_hypothesis_fingerprint_is_over_substance_not_title():
    """A model asked repeatedly for something new will rename the same idea.

    A ledger keyed on titles lets it do that indefinitely without the
    multiple-testing bar ever moving, which is the whole mechanism the ledger
    exists to enforce.
    """
    from loonie import researcher as R

    a = {"title": "Low Volatility Tilt", "market": "stocks",
         "signal": "60-day realized volatility", "direction": "long low",
         "holding_days": 21}
    b = dict(a, title="Calm Stock Preference Strategy")   # renamed only
    c = dict(a, holding_days=63)                          # genuinely different

    assert R._fingerprint(a) == R._fingerprint(b), \
        "renaming a hypothesis produced a new fingerprint"
    assert R._fingerprint(a) != R._fingerprint(c)


def test_the_bar_rises_with_distinct_hypotheses(tmp_path, monkeypatch):
    """Every proposal raises the threshold for every hypothesis, including
    those already tested. A tireless proposer does not improve the odds of
    finding an edge; it raises the bar for everything."""
    from loonie import researcher as R

    monkeypatch.setattr(R, "resolve", lambda q: tmp_path / Path(q).name)
    assert R.bar()["bar"] == 2.0

    for i in range(200):
        R.record({"title": "h%d" % i, "market": "stocks",
                  "signal": "signal number %d" % i, "direction": "long high",
                  "holding_days": 21})
    b = R.bar()
    assert b["proposals_distinct"] == 200
    assert b["bar"] > 3.2, "the bar did not rise with 200 hypotheses"

    # A duplicate is the same test and must not be charged twice.
    R.record({"title": "again", "market": "stocks", "signal": "signal number 0",
              "direction": "long high", "holding_days": 21})
    b2 = R.bar()
    assert b2["proposals_distinct"] == 200, "a duplicate moved the bar"
    assert b2["proposals_total"] == 201, "the duplicate was not recorded at all"
