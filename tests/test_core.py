"""Tests for the properties that, if broken, make every number meaningless.

The lookahead tests are the important ones. A backtest with a one-bar leak
does not look broken -- it looks brilliant, which is much worse.

    python -m pytest tests/ -v
"""
from __future__ import annotations

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
                 ("min_null_percentile", 0.0), ("min_stress_alpha", -1e9)):
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
