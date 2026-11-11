"""Reproduce Member 3 Phase 2.4 statistical validation results.

Run from the repository root:
    python reports/report4/member3_phase2_4/reproduce_phase2_4.py --output-dir reports/report4/member3_phase2_4/reproduction_run_YYYYMMDD_HHMMSS

Pass ``--output-dir`` with a new, non-existing directory. Existing outputs
are never overwritten.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
import warnings
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import scipy
import statsmodels
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy.stats import chi2_contingency, kruskal, rankdata
from statsmodels.stats.multitest import multipletests
from statsmodels.genmod.cov_struct import Independence

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from src.utils.supabase_client import DEFAULT_SELECT_COLUMNS


EXPECTED_FINGERPRINT = "bb0aea6d42049da728c8339bbf92388991825d9128cb85122df081e71a0a6e81"
FEATURES = [
    "word_count", "unique_word_count", "lexical_diversity",
    "repetition_ratio", "avg_word_length", "chorus_repetition_count",
]
EXPECTED_KW = {
    "word_count": (487, 721, 532, 269.884707835, 2.4847363792868758e-59, 1.242368189643438e-58, 0.15422263),
    "unique_word_count": (487, 721, 532, 135.783969165, 3.272545777885278e-30, 6.545091555770556e-30, 0.07702013),
    "lexical_diversity": (487, 721, 532, 147.713946188, 8.400867502165252e-33, 3.3603470008661007e-32, 0.08388828),
    "repetition_ratio": (487, 721, 532, 147.713946188, 8.400867502165252e-33, 3.3603470008661007e-32, 0.08388828),
    "avg_word_length": (487, 721, 532, 557.065993993, 1.083069626315294e-121, 6.498417757891764e-121, 0.31955440),
    "chorus_repetition_count": (460, 629, 512, 2.368122346, 0.3060333570, 0.3060333570, 0.00023036),
}
EXPECTED_AUDIO = (24.3141791037, 2, 5.2510129777e-6, 0.11810849)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def canonical_fingerprint(df: pd.DataFrame) -> str:
    sort_cols = [c for c in ["spotify_track_id", "market"] if c in df.columns]
    ordered = df.sort_values(sort_cols).copy()
    cols = [c for c in DEFAULT_SELECT_COLUMNS if c in ordered.columns]
    payload = ordered[cols].copy()
    for col in payload.select_dtypes(include=["floating", "float"]).columns:
        payload[col] = payload[col].round(6)
    return hashlib.sha256(payload.to_csv(index=False).encode("utf-8")).hexdigest()


def assert_close(label: str, actual: float, expected: float, *, atol: float = 1e-7, rtol: float = 1e-5) -> bool:
    ok = bool(np.isclose(actual, expected, atol=atol, rtol=rtol, equal_nan=False))
    CHECKS.append((label, ok, f"actual={actual:.12g}; expected={expected:.12g}"))
    return ok


CHECKS: list[tuple[str, bool, str]] = []


def wald_market(result) -> tuple[float, int, float]:
    names = list(result.params.index)
    indices = [i for i, name in enumerate(names) if name != "Intercept"]
    r = np.zeros((len(indices), len(names)))
    for row, col in enumerate(indices):
        r[row, col] = 1.0
    test = result.wald_test((r, np.zeros(len(indices))), scalar=True)
    return float(test.statistic), len(indices), float(test.pvalue)


def gee_diagnostics(result) -> tuple[int, bool, int]:
    """Return fit iterations and robust market-contrast covariance checks."""
    iterations = len(result.fit_history.get("params", []))
    cov = np.asarray(result.cov_params(), dtype=float)
    terms = [i for i, name in enumerate(result.params.index) if name != "Intercept"]
    contrast_cov = cov[np.ix_(terms, terms)]
    return iterations, bool(np.isfinite(cov).all()), int(np.linalg.matrix_rank(contrast_cov))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-dir", required=True,
        help="New output directory; must not already exist.",
    )
    args = parser.parse_args()
    out = (ROOT / args.output_dir).resolve()
    if out.exists():
        print(f"FAIL: refusing to overwrite existing output directory: {out}", file=sys.stderr)
        return 2

    paths = {
        "cache": ROOT / "data/processed/modeling_dataset_clean_cache.csv",
        "features": ROOT / "reports/report3/lyrics_features_extracted.csv",
        "provenance": ROOT / "data/raw/lyrics_collection_results.json",
        "historical_mode": ROOT / "reports/report4/chart_data/member2/fig3_tonal_harmony_market.csv",
        "historical_rows": ROOT / "reports/report4/chart_data/member2/audio_features_eda_source.csv",
    }
    missing_paths = [str(p.relative_to(ROOT)) for p in paths.values() if not p.is_file()]
    if missing_paths:
        print("FAIL: required inputs missing: " + ", ".join(missing_paths), file=sys.stderr)
        return 2

    cache = pd.read_csv(paths["cache"])
    fp = canonical_fingerprint(cache)
    if fp != EXPECTED_FINGERPRINT:
        print(f"STOP: canonical fingerprint mismatch: {fp}", file=sys.stderr)
        return 3
    CHECKS.append(("canonical fingerprint", True, fp))
    if len(cache) != 2098 or cache.spotify_track_id.nunique() != 1912:
        print("STOP: canonical row or track count changed", file=sys.stderr)
        return 3
    CHECKS.extend([
        ("canonical market rows", len(cache) == 2098, str(len(cache))),
        ("canonical unique tracks", cache.spotify_track_id.nunique() == 1912, str(cache.spotify_track_id.nunique())),
        ("canonical pair uniqueness", not cache.duplicated(["spotify_track_id", "market"]).any(), str(cache.duplicated(["spotify_track_id", "market"]).sum())),
    ])

    features = pd.read_csv(paths["features"])
    provenance = json.loads(paths["provenance"].read_text(encoding="utf-8"))
    if features.spotify_track_id.nunique() != len(features) or len(features) != 1912:
        raise ValueError("Extracted feature table must have one row per each of 1,912 track IDs")
    if features["spotify_track_id"].isna().any():
        raise ValueError("Missing spotify_track_id in extracted feature table")

    expanded = {
        (track_id, item["market"])
        for track_id, record in provenance.items()
        for item in record.get("sampling_provenance", [])
    }
    cache_pairs = set(zip(cache.spotify_track_id, cache.market))
    if expanded != cache_pairs:
        raise ValueError(f"Provenance/cache pair mismatch: {len(expanded ^ cache_pairs)} differing pairs")
    if set(features.spotify_track_id) != set(provenance):
        raise ValueError("Feature/provenance track IDs do not match")

    # Every market pair inherits the extracted track-level NLP values, and
    # carries audio_mode from that exact canonical track-market cache row.
    track_cols = ["spotify_track_id", "lyrics_status", *FEATURES]
    track_frame = features[track_cols].copy()
    expanded_rows = [(tid, market) for tid, market in sorted(expanded)]
    market = pd.DataFrame(expanded_rows, columns=["spotify_track_id", "market"])
    market = market.merge(track_frame, on="spotify_track_id", how="left", validate="many_to_one")
    market = market.merge(
        cache[["spotify_track_id", "market", "audio_mode"]],
        on=["spotify_track_id", "market"], how="outer", validate="one_to_one", indicator=True,
    )
    if not market["_merge"].eq("both").all():
        raise ValueError("Market expansion does not exactly match canonical cache keys")
    market.drop(columns="_merge", inplace=True)
    market["market"] = pd.Categorical(market.market, categories=["US", "VN", "KR"])
    counts_by_track = market.groupby("spotify_track_id", observed=True).market.nunique()
    market["exclusive"] = market.spotify_track_id.map(counts_by_track).eq(1)
    exclusive = market.loc[market.exclusive].copy()
    if len(exclusive) != 1743 or exclusive.spotify_track_id.nunique() != 1743:
        raise ValueError(f"Exclusive subset changed: rows={len(exclusive)} tracks={exclusive.spotify_track_id.nunique()}")
    CHECKS.extend([
        ("expanded rows", len(market) == 2098, str(len(market))),
        ("exclusive tracks", len(exclusive) == 1743 and exclusive.spotify_track_id.nunique() == 1743, str(exclusive.spotify_track_id.nunique())),
        ("no duplicate expanded pairs", not market.duplicated(["spotify_track_id", "market"]).any(), str(market.duplicated(["spotify_track_id", "market"]).sum())),
    ])

    # Verify the duplicated feature representations that exist in the cache.
    audit_mismatches = {}
    for col in ["word_count", "repetition_ratio", "avg_word_length", "chorus_repetition_count"]:
        joined = market.merge(cache[["spotify_track_id", "market", col]], on=["spotify_track_id", "market"], suffixes=("_lyrics", "_cache"), validate="one_to_one")
        left, right = joined[f"{col}_lyrics"], joined[f"{col}_cache"]
        same = pd.Series(np.isclose(
            pd.to_numeric(left, errors="coerce").to_numpy(dtype=float),
            pd.to_numeric(right, errors="coerce").to_numpy(dtype=float),
            rtol=1e-12, atol=1e-12, equal_nan=True,
        ), index=joined.index)
        audit_mismatches[col] = int((~same).sum())
        if audit_mismatches[col]:
            raise ValueError(f"Extracted/cache {col} mismatch on {audit_mismatches[col]} pairs")

    # Primary Kruskal-Wallis analyses: complete cases per feature, exclusive tracks.
    kw_rows = []
    raw_ps = []
    for feature in FEATURES:
        groups = [exclusive.loc[exclusive.market == m, feature].dropna().astype(float).to_numpy() for m in ["VN", "US", "KR"]]
        h, p = kruskal(*groups)
        ns = [len(g) for g in groups]
        n_total = sum(ns)
        eps = float((h - len(groups) + 1) / (n_total - len(groups)))
        kw_rows.append({
            "analysis_type": "primary", "observational_unit": "market-exclusive track",
            "feature": feature, "n_vn": ns[0], "n_us": ns[1], "n_kr": ns[2],
            "n_total": n_total, "h_statistic": float(h), "df": 2,
            "p_raw": float(p), "epsilon_squared": eps,
        })
        raw_ps.append(float(p))
    adjusted = multipletests(raw_ps, alpha=0.05, method="holm")[1]
    for row, p_adj in zip(kw_rows, adjusted):
        row["p_holm"] = float(p_adj)
        row["alpha"] = 0.05
        row["significant_holm"] = bool(p_adj < 0.05)
    for feature, row in zip(FEATURES, kw_rows):
        expected = EXPECTED_KW[feature]
        observed = (row["n_vn"], row["n_us"], row["n_kr"], row["h_statistic"], row["p_raw"], row["p_holm"], row["epsilon_squared"])
        for label, actual, target in zip(["n_vn", "n_us", "n_kr", "H", "p_raw", "p_holm", "epsilon_squared"], observed, expected):
            if label.startswith("n_"):
                CHECKS.append((f"KW {feature} {label}", actual == target, f"actual={actual}; expected={target}"))
            elif label.startswith("p_"):
                assert_close(f"KW {feature} {label}", float(actual), float(target), atol=1e-300, rtol=0.025)
            else:
                assert_close(f"KW {feature} {label}", float(actual), float(target), atol=5e-6, rtol=2e-4)

    # Primary Pearson Chi-Square: exclusive tracks only.
    audio_tab = pd.crosstab(exclusive.market, exclusive.audio_mode).reindex(index=["VN", "US", "KR"], columns=["major", "minor"], fill_value=0)
    audio_res = chi2_contingency(audio_tab, correction=False)
    audio_v = float(np.sqrt(audio_res.statistic / (audio_tab.values.sum() * min(audio_tab.shape[0] - 1, audio_tab.shape[1] - 1))))
    assert_close("primary audio chi-square", float(audio_res.statistic), EXPECTED_AUDIO[0], atol=5e-6)
    CHECKS.append(("primary audio df", audio_res.dof == EXPECTED_AUDIO[1], str(audio_res.dof)))
    assert_close("primary audio p", float(audio_res.pvalue), EXPECTED_AUDIO[2], atol=1e-300, rtol=0.025)
    assert_close("primary audio Cramer's V", audio_v, EXPECTED_AUDIO[3], atol=5e-6)
    audio_rows = []
    for i, m in enumerate(["VN", "US", "KR"]):
        total = int(audio_tab.loc[m].sum())
        for j, mode in enumerate(["major", "minor"]):
            audio_rows.append({
                "analysis_type": "primary", "observational_unit": "market-exclusive track",
                "market": m, "audio_mode": mode, "observed": int(audio_tab.loc[m, mode]),
                "expected": float(audio_res.expected_freq[i, j]), "market_total": total,
                "market_percent": float(audio_tab.loc[m, mode] / total * 100),
                "chi_square": float(audio_res.statistic), "df": int(audio_res.dof),
                "p_raw": float(audio_res.pvalue), "cramers_v": audio_v, "alpha": 0.05,
            })

    # Chorus zero-fill sensitivity: exclude missing-lyrics tracks before filling.
    zero_base = exclusive.loc[exclusive.lyrics_status.ne("not_found")].copy()
    zero_missing = zero_base.chorus_repetition_count.isna()
    zero_by_market = zero_base.loc[zero_missing].groupby("market", observed=True).size().reindex(["VN", "US", "KR"], fill_value=0)
    zero_base["chorus_repetition_count"] = zero_base.chorus_repetition_count.fillna(0)
    zero_groups = [zero_base.loc[zero_base.market == m, "chorus_repetition_count"].astype(float).to_numpy() for m in ["VN", "US", "KR"]]
    zero_h, zero_p = kruskal(*zero_groups)
    zero_n = sum(map(len, zero_groups))
    zero_eps = float((zero_h - 2) / (zero_n - 3))
    zero_rows = [{"analysis_type": "sensitivity", "row_type": "market", "market": m,
                  "n": len(zero_groups[i]), "zero_filled_n": int(zero_by_market.loc[m]),
                  "h_statistic": float(zero_h), "df": 2, "p_raw": float(zero_p),
                  "epsilon_squared": zero_eps, "assumption": "unidentifiable missing chorus count treated as zero"}
                 for i, m in enumerate(["VN", "US", "KR"])]
    zero_rows.append({"analysis_type": "sensitivity", "row_type": "overall", "market": "ALL",
                      "n": zero_n, "zero_filled_n": int(zero_missing.sum()), "h_statistic": float(zero_h),
                      "df": 2, "p_raw": float(zero_p), "epsilon_squared": zero_eps,
                      "assumption": "unidentifiable missing chorus count treated as zero"})

    # All-market rank-normalized GEE sensitivity for NLP features.
    gee_rows, gee_warnings = [], []
    for feature in FEATURES:
        data = market[["spotify_track_id", "market", feature]].dropna().copy()
        data["rank_normalized"] = rankdata(data[feature].astype(float), method="average") / (len(data) + 1)
        data["market"] = pd.Categorical(data.market, categories=["US", "VN", "KR"])
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            fit = smf.gee(
                "rank_normalized ~ C(market, Treatment(reference='US'))",
                groups="spotify_track_id", data=data,
                family=sm.families.Gaussian(), cov_struct=Independence(),
            ).fit(cov_type="robust")
        gee_warnings.extend(f"{feature}: {w.message}" for w in caught)
        statistic, df, p = wald_market(fit)
        iterations, cov_finite, contrast_rank = gee_diagnostics(fit)
        gee_rows.append({
            "analysis_type": "sensitivity", "feature": feature, "rows": len(data),
            "clusters": data.spotify_track_id.nunique(), "wald_chi_square": statistic,
            "df": df, "p_raw": p, "converged": bool(fit.converged),
            "fit_iterations": iterations, "robust_covariance_finite": cov_finite,
            "market_contrast_covariance_rank": contrast_rank,
            "covariance": "robust sandwich", "working_correlation": "independence",
            "rank_transform": "average rank / (N + 1)",
            "vn_vs_us_coefficient": float(fit.params.iloc[1]),
            "kr_vs_us_coefficient": float(fit.params.iloc[2]),
        })
    gee_adj = multipletests([r["p_raw"] for r in gee_rows], method="holm")[1]
    for row, p_adj in zip(gee_rows, gee_adj):
        row["p_holm"] = float(p_adj)

    # All-market logistic GEE for audio_mode (major=1).
    audio_gee_data = market[["spotify_track_id", "market", "audio_mode"]].dropna().copy()
    audio_gee_data["major"] = audio_gee_data.audio_mode.eq("major").astype(int)
    audio_gee_data["market"] = pd.Categorical(audio_gee_data.market, categories=["US", "VN", "KR"])
    audio_gee_fit, audio_gee_warnings = fit_gee(
        "major ~ C(market, Treatment(reference='US'))", audio_gee_data,
        "spotify_track_id", sm.families.Binomial(),
    )
    audio_gee_stat, audio_gee_df, audio_gee_p = wald_market(audio_gee_fit)
    audio_gee_iterations, audio_gee_cov_finite, audio_gee_contrast_rank = gee_diagnostics(audio_gee_fit)
    audio_gee_rows = [{
        "analysis_type": "sensitivity", "term": term,
        "rows": len(audio_gee_data), "clusters": audio_gee_data.spotify_track_id.nunique(),
        "coefficient_log_odds": float(audio_gee_fit.params[term]),
        "odds_ratio": float(np.exp(audio_gee_fit.params[term])),
        "coefficient_p": float(audio_gee_fit.pvalues[term]),
        "omnibus_wald_chi_square": audio_gee_stat, "omnibus_df": audio_gee_df,
        "omnibus_p": audio_gee_p, "converged": bool(audio_gee_fit.converged),
        "fit_iterations": audio_gee_iterations, "robust_covariance_finite": audio_gee_cov_finite,
        "market_contrast_covariance_rank": audio_gee_contrast_rank,
        "covariance": "robust sandwich", "working_correlation": "independence",
    } for term in audio_gee_fit.params.index if term != "Intercept"]

    # Missingness counts, exclusive and all-market Pearson tables, and clustered GEE.
    missingness_rows = []
    miss_tabs = {}
    for label, frame, kind in [("exclusive", exclusive, "primary independent Pearson"),
                               ("all_market", market, "diagnostic only; repeated tracks")]:
        tab = pd.crosstab(frame.market, frame.chorus_repetition_count.isna()).reindex(index=["VN", "US", "KR"], columns=[False, True], fill_value=0)
        res = chi2_contingency(tab, correction=False)
        miss_tabs[label] = (tab, res)
        for i, m in enumerate(["VN", "US", "KR"]):
            total = int(tab.loc[m].sum())
            for j, status in enumerate(["not_missing", "missing"]):
                missingness_rows.append({
                    "analysis_type": kind, "observational_unit": "track" if label == "exclusive" else "track-market row",
                    "market": m, "missing_status": status,
                    "observed": int(tab.loc[m, j == 1]), "expected": float(res.expected_freq[i, j]),
                    "market_total": total, "missing_percent": float(tab.loc[m, True] / total * 100),
                    "chi_square": float(res.statistic), "df": int(res.dof), "p_raw": float(res.pvalue),
                    "min_expected": float(res.expected_freq.min()),
                })
    gee_missing_data = market[["spotify_track_id", "market", "chorus_repetition_count"]].copy()
    gee_missing_data["missing_chorus"] = gee_missing_data.chorus_repetition_count.isna().astype(int)
    gee_missing_data["market"] = pd.Categorical(gee_missing_data.market, categories=["US", "VN", "KR"])
    missing_fit, missing_warnings = fit_gee(
        "missing_chorus ~ C(market, Treatment(reference='US'))", gee_missing_data,
        "spotify_track_id", sm.families.Binomial(),
    )
    miss_stat, miss_df, miss_p = wald_market(missing_fit)
    missing_iterations, missing_cov_finite, missing_contrast_rank = gee_diagnostics(missing_fit)
    for term in missing_fit.params.index:
        if term == "Intercept":
            continue
        missingness_rows.append({
            "analysis_type": "sensitivity; cluster-aware logistic GEE",
            "observational_unit": "track-market row; cluster=spotify_track_id",
            "market": term, "missing_status": "market coefficient vs US",
            "observed": np.nan, "expected": np.nan, "market_total": len(gee_missing_data),
            "missing_percent": np.nan, "chi_square": miss_stat, "df": miss_df,
            "p_raw": miss_p, "min_expected": np.nan,
            "coefficient_log_odds": float(missing_fit.params[term]),
            "odds_ratio": float(np.exp(missing_fit.params[term])),
            "coefficient_p": float(missing_fit.pvalues[term]),
            "clusters": gee_missing_data.spotify_track_id.nunique(),
            "converged": bool(missing_fit.converged),
            "fit_iterations": missing_iterations, "robust_covariance_finite": missing_cov_finite,
            "market_contrast_covariance_rank": missing_contrast_rank,
            "covariance": "robust sandwich", "working_correlation": "independence",
        })

    # Historical Report 4 source reconciliation; cited aggregation and raw rows.
    historical = pd.read_csv(paths["historical_mode"])
    hist_tab = historical.groupby(["market", "audio_mode"])["count"].sum().unstack(fill_value=0).reindex(index=["VN", "US", "KR"], columns=["major", "minor"], fill_value=0)
    hist_res = chi2_contingency(hist_tab, correction=False)
    historical_rows = []
    for i, m in enumerate(["VN", "US", "KR"]):
        for j, mode in enumerate(["major", "minor"]):
            historical_rows.append({
                "analysis_type": "historical reconciliation",
                "observational_unit": "historical market row; repeated tracks present",
                "market": m, "audio_mode": mode, "observed": int(hist_tab.loc[m, mode]),
                "expected": float(hist_res.expected_freq[i, j]), "market_total": int(hist_tab.loc[m].sum()),
                "chi_square": float(hist_res.statistic), "df": int(hist_res.dof), "p_raw": float(hist_res.pvalue),
                "reported_report4_p": 0.42, "v41_target_chi_square": 0.51, "v41_target_p": 0.774,
                "reconciliation": "p=0.42 not reproduced from cited source; v4.1 source/method unavailable",
            })
    historical_raw = pd.read_csv(paths["historical_rows"])
    raw_hist_tab = pd.crosstab(historical_raw.market, historical_raw.audio_mode).reindex(index=["VN", "US", "KR"], columns=["major", "minor"], fill_value=0)
    CHECKS.append(("historical aggregate equals row source", hist_tab.equals(raw_hist_tab), f"rows={len(historical_raw)}"))

    # Track-level / market-row missingness reconciliation.
    nmarkets = market.groupby("spotify_track_id", observed=True).market.nunique()
    track_n = features.spotify_track_id.nunique()
    track_missing = int(features.chorus_repetition_count.isna().sum())
    all_missing = int(market.chorus_repetition_count.isna().sum())
    exclusive_missing = int(exclusive.chorus_repetition_count.isna().sum())
    no_lyrics_ids = set(features.loc[features.lyrics_status.eq("not_found"), "spotify_track_id"])
    miss_ids = set(features.loc[features.chorus_repetition_count.isna(), "spotify_track_id"])
    no_lyrics_rows = int(market.spotify_track_id.isin(no_lyrics_ids).sum())
    zero_filled_all = all_missing - no_lyrics_rows
    reconciliation = pd.DataFrame([
        {"analysis_type": "descriptive reconciliation", "observational_unit": "unique track", "total_n": track_n,
         "missing_chorus_n": track_missing, "vn_missing": int(features.loc[features.spotify_track_id.isin(miss_ids & set(market.loc[market.market == 'VN', 'spotify_track_id'])), 'spotify_track_id'].nunique()),
         "us_missing": int(features.loc[features.spotify_track_id.isin(miss_ids & set(market.loc[market.market == 'US', 'spotify_track_id'])), 'spotify_track_id'].nunique()),
         "kr_missing": int(features.loc[features.spotify_track_id.isin(miss_ids & set(market.loc[market.market == 'KR', 'spotify_track_id'])), 'spotify_track_id'].nunique()),
         "includes_missing_lyrics": True, "note": "market membership counts overlap for repeated tracks"},
        {"analysis_type": "descriptive reconciliation", "observational_unit": "all-market track-market row", "total_n": len(market),
         "missing_chorus_n": all_missing, "vn_missing": int(market.loc[market.market == 'VN', 'chorus_repetition_count'].isna().sum()),
         "us_missing": int(market.loc[market.market == 'US', 'chorus_repetition_count'].isna().sum()),
         "kr_missing": int(market.loc[market.market == 'KR', 'chorus_repetition_count'].isna().sum()),
         "includes_missing_lyrics": True, "note": f"{no_lyrics_rows} missing-lyrics rows included"},
        {"analysis_type": "descriptive reconciliation", "observational_unit": "market-exclusive track", "total_n": len(exclusive),
         "missing_chorus_n": exclusive_missing, "vn_missing": int(exclusive.loc[exclusive.market == 'VN', 'chorus_repetition_count'].isna().sum()),
         "us_missing": int(exclusive.loc[exclusive.market == 'US', 'chorus_repetition_count'].isna().sum()),
         "kr_missing": int(exclusive.loc[exclusive.market == 'KR', 'chorus_repetition_count'].isna().sum()),
         "includes_missing_lyrics": True, "note": "one row per track"},
    ])

    provenance_rows = []
    for name, path in paths.items():
        provenance_rows.append({"record_type": "input", "name": name, "path": str(path.relative_to(ROOT)), "sha256": sha256_file(path), "rows": None, "value": None})
    for name, value in [
        ("canonical_logical_fingerprint", fp), ("canonical_market_rows", len(cache)),
        ("canonical_unique_tracks", cache.spotify_track_id.nunique()), ("exclusive_tracks", len(exclusive)),
        ("canonical_duplicate_track_market_pairs", int(cache.duplicated(["spotify_track_id", "market"]).sum())),
        ("feature_track_rows", len(features)), ("provenance_unique_market_pairs", len(expanded)),
    ]:
        provenance_rows.append({"record_type": "audit", "name": name, "path": "", "sha256": "", "rows": None, "value": value})
    provenance = pd.DataFrame(provenance_rows)

    # Critical checkpoint comparisons for sensitivity outputs.
    # Use approved values only as validation targets, never in calculations.
    expected_gee = {
        "word_count": (2095, 1909, 290.0772, 1.02458e-63, 5.12290e-63),
        "unique_word_count": (2095, 1909, 128.0916, 1.53204e-28, 6.12818e-28),
        "lexical_diversity": (2095, 1909, 108.24176, 3.1304e-24, 9.39121e-24),
        "repetition_ratio": (2095, 1909, 108.24176, 3.1304e-24, 9.39121e-24),
        "avg_word_length": (2095, 1909, 772.352, 1.9315e-168, 1.1589e-167),
        "chorus_repetition_count": (1943, 1764, 3.33091, 0.189104, 0.189104),
    }
    gee_adj_check = multipletests([r["p_raw"] for r in gee_rows], method="holm")[1]
    for row, p_adj in zip(gee_rows, gee_adj_check):
        row["p_holm"] = float(p_adj)
    for row in gee_rows:
        exp = expected_gee[row["feature"]]
        CHECKS.append((f"GEE {row['feature']} rows/clusters", (row["rows"], row["clusters"]) == exp[:2], f"actual={(row['rows'], row['clusters'])}; expected={exp[:2]}"))
        assert_close(f"GEE {row['feature']} Wald", row["wald_chi_square"], exp[2], atol=0.02)
        assert_close(f"GEE {row['feature']} p", row["p_raw"], exp[3], atol=abs(exp[3]) * 0.03 + 1e-300, rtol=0.03)
        assert_close(f"GEE {row['feature']} Holm p", row["p_holm"], exp[4], atol=abs(exp[4]) * 0.03 + 1e-300, rtol=0.03)
        CHECKS.append((f"GEE {row['feature']} converged", row["converged"], str(row["converged"])))
        CHECKS.append((f"GEE {row['feature']} iterations", row["fit_iterations"] == 2, str(row["fit_iterations"])))
        CHECKS.append((f"GEE {row['feature']} robust covariance", row["robust_covariance_finite"] and row["market_contrast_covariance_rank"] == 2, f"finite={row['robust_covariance_finite']}; contrast_rank={row['market_contrast_covariance_rank']}"))
    CHECKS.append(("audio logistic GEE converged", bool(audio_gee_fit.converged), str(audio_gee_fit.converged)))
    CHECKS.append(("audio logistic GEE iterations", audio_gee_iterations == 2, str(audio_gee_iterations)))
    audio_terms = {
        "vn": "C(market, Treatment(reference='US'))[T.VN]",
        "kr": "C(market, Treatment(reference='US'))[T.KR]",
    }
    CHECKS.append(("audio logistic GEE rows/clusters", (len(audio_gee_data), audio_gee_data.spotify_track_id.nunique()) == (2098, 1912), f"actual={(len(audio_gee_data), audio_gee_data.spotify_track_id.nunique())}"))
    CHECKS.append(("audio logistic GEE df", audio_gee_df == 2, str(audio_gee_df)))
    assert_close("audio logistic GEE omnibus Wald", audio_gee_stat, 21.723163, atol=5e-5, rtol=2e-5)
    assert_close("audio logistic GEE omnibus p", audio_gee_p, 1.918e-5, atol=1e-300, rtol=0.025)
    for label, term, coef, odds in [
        ("VN vs US", audio_terms["vn"], 0.498085, 1.645567),
        ("KR vs US", audio_terms["kr"], 0.116763, 1.123853),
    ]:
        assert_close(f"audio logistic GEE {label} coefficient", float(audio_gee_fit.params[term]), coef, atol=2e-5, rtol=2e-5)
        assert_close(f"audio logistic GEE {label} OR", float(np.exp(audio_gee_fit.params[term])), odds, atol=2e-5, rtol=2e-5)
    CHECKS.append(("audio logistic GEE robust covariance", audio_gee_cov_finite and audio_gee_contrast_rank == 2, f"finite={audio_gee_cov_finite}; contrast_rank={audio_gee_contrast_rank}"))
    CHECKS.append(("missingness logistic GEE converged", bool(missing_fit.converged), str(missing_fit.converged)))
    CHECKS.append(("missingness logistic GEE iterations", missing_iterations == 2, str(missing_iterations)))
    CHECKS.append(("missingness logistic GEE rows/clusters", (len(gee_missing_data), gee_missing_data.spotify_track_id.nunique()) == (2098, 1912), f"actual={(len(gee_missing_data), gee_missing_data.spotify_track_id.nunique())}"))
    CHECKS.append(("missingness logistic GEE df", miss_df == 2, str(miss_df)))
    assert_close("missingness logistic GEE omnibus Wald", miss_stat, 36.034639, atol=5e-5, rtol=2e-5)
    assert_close("missingness logistic GEE omnibus p", miss_p, 1.496847e-8, atol=1e-300, rtol=0.025)
    for label, term, coef, odds in [
        ("VN vs US", "C(market, Treatment(reference='US'))[T.VN]", -0.857806, 0.424092),
        ("KR vs US", "C(market, Treatment(reference='US'))[T.KR]", -1.205396, 0.299573),
    ]:
        assert_close(f"missingness logistic GEE {label} coefficient", float(missing_fit.params[term]), coef, atol=2e-5, rtol=2e-5)
        assert_close(f"missingness logistic GEE {label} OR", float(np.exp(missing_fit.params[term])), odds, atol=2e-5, rtol=2e-5)
    CHECKS.append(("missingness logistic GEE robust covariance", missing_cov_finite and missing_contrast_rank == 2, f"finite={missing_cov_finite}; contrast_rank={missing_contrast_rank}"))
    if gee_warnings or audio_gee_warnings or missing_warnings:
        CHECKS.append(("GEE no warnings", False, " | ".join(gee_warnings + audio_gee_warnings + missing_warnings)))
    else:
        CHECKS.append(("GEE no warnings", True, "none"))

    # Audit final validations; stop before creating outputs if a checkpoint fails.
    failed = [c for c in CHECKS if not c[1]]
    if failed:
        print("VALIDATION FAILURES:")
        for name, _, detail in failed:
            print(f"- {name}: {detail}")
        return 4

    out.mkdir(parents=True, exist_ok=False)
    pd.DataFrame(kw_rows).to_csv(out / "01_primary_kruskal_wallis.csv", index=False)
    pd.DataFrame(audio_rows).to_csv(out / "02_primary_audio_mode_chi_square.csv", index=False)
    pd.DataFrame(zero_rows).to_csv(out / "03_chorus_zero_fill_sensitivity.csv", index=False)
    pd.DataFrame(gee_rows).to_csv(out / "04_all_market_rank_gee_sensitivity.csv", index=False)
    pd.DataFrame(audio_gee_rows).to_csv(out / "05_audio_mode_logistic_gee.csv", index=False)
    reconciliation.to_csv(out / "06_chorus_missingness_reconciliation.csv", index=False)
    pd.DataFrame(missingness_rows).to_csv(out / "07_missingness_association.csv", index=False)
    pd.DataFrame(historical_rows).to_csv(out / "08_historical_audio_mode_reconciliation.csv", index=False)
    provenance.to_csv(out / "09_data_provenance_audit.csv", index=False)

    import subprocess
    git_branch = subprocess.run(["git", "branch", "--show-current"], cwd=ROOT, text=True, capture_output=True).stdout.strip()
    git_commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True, capture_output=True).stdout.strip()
    git_status = subprocess.run(["git", "status", "--short"], cwd=ROOT, text=True, capture_output=True).stdout.strip() or "clean"
    package_versions = {
        "python": platform.python_version(), "pandas": pd.__version__, "numpy": np.__version__,
        "scipy": scipy.__version__, "statsmodels": statsmodels.__version__,
    }
    run_record = {
        "run_utc": datetime.now(timezone.utc).isoformat(), "git_branch": git_branch,
        "git_commit": git_commit, "git_status_at_log_capture": git_status,
        "python_packages": package_versions,
        "input_paths_and_sha256": {name: {"path": str(path.relative_to(ROOT)), "sha256": sha256_file(path)} for name, path in paths.items()},
        "canonical_logical_fingerprint": fp,
        "counts": {"market_rows": len(cache), "unique_tracks": cache.spotify_track_id.nunique(),
                   "exclusive_tracks": len(exclusive), "expanded_pairs": len(expanded)},
        "warnings": gee_warnings + audio_gee_warnings + missing_warnings,
        "limitations": [
            "Exclusive-track inference applies only to songs appearing in exactly one market and may be affected by selection bias.",
            "Zero-fill chorus sensitivity assumes unidentifiable chorus counts equal zero; it is not primary.",
            "GEE and Kruskal-Wallis target different estimands.",
            "Historical Report 4 p=0.42 is not reproduced; v4.1 source/method remains unidentified.",
            "Project requirements specify SciPy 1.18.1; run environment SciPy version is recorded and no package was installed.",
        ],
        "validations": [{"name": n, "passed": bool(ok), "detail": detail} for n, ok, detail in CHECKS],
        "results": {
            "primary_kw": kw_rows,
            "primary_audio_mode": {"observed": audio_tab.to_dict(), "chi_square": float(audio_res.statistic), "df": int(audio_res.dof), "p": float(audio_res.pvalue), "cramers_v": audio_v},
            "chorus_zero_fill": {"market_n": {m: len(g) for m, g in zip(["VN", "US", "KR"], zero_groups)}, "zero_filled": zero_by_market.to_dict(), "H": float(zero_h), "p": float(zero_p), "epsilon_squared": zero_eps},
            "rank_gee": gee_rows, "audio_logistic_gee": audio_gee_rows,
            "missingness_logistic_gee": {"rows": len(gee_missing_data), "clusters": gee_missing_data.spotify_track_id.nunique(), "wald_chi_square": miss_stat, "df": miss_df, "p": miss_p, "converged": bool(missing_fit.converged), "fit_iterations": missing_iterations, "robust_covariance_finite": missing_cov_finite, "market_contrast_covariance_rank": missing_contrast_rank},
            "historical_audio_mode": {"observed": hist_tab.to_dict(), "chi_square": float(hist_res.statistic), "df": int(hist_res.dof), "p": float(hist_res.pvalue)},
        },
    }
    (out / "run_log.json").write_text(json.dumps(run_record, indent=2, allow_nan=False), encoding="utf-8")
    print("Validation PASS")
    for name, ok, detail in CHECKS:
        print(f"{'PASS' if ok else 'FAIL'} | {name} | {detail}")
    print(f"Outputs: {out.relative_to(ROOT)}")
    return 0


def fit_gee(formula: str, data: pd.DataFrame, group: str, family):
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result = smf.gee(formula, groups=group, data=data, family=family, cov_struct=Independence()).fit(cov_type="robust")
    return result, [str(w.message) for w in caught]


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"VALIDATION ERROR: {type(exc).__name__}: {exc}", file=sys.stderr)
        raise
