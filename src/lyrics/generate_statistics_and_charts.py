#!/usr/bin/env python3
"""Generate descriptive statistics, chart data CSVs, publication-grade figures,
and Member 1 / Member 4 handoff datasets for Member 3 (Stages 7 & 9).
"""
from __future__ import annotations

import csv
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

ROOT = Path(__file__).resolve().parents[2]
FEATURES_CSV = ROOT / "reports/report3/lyrics_features_extracted.csv"
INVENTORY_CSV = ROOT / "reports/report3/lyrics_inventory.csv"
RESULTS_JSON = ROOT / "data/raw/lyrics_collection_results.json"

STATS_CSV = ROOT / "reports/report3/lyrics_statistics.csv"
FIGURES_DIR = ROOT / "reports/report4/figures/member3"
CHART_DATA_DIR = ROOT / "reports/report4/chart_data/member3"

HANDOFF_M1_CSV = ROOT / "reports/report3/handoff_member1_chart_join.csv"
HANDOFF_M4_VALID_CSV = ROOT / "reports/report3/handoff_member4_valid_tracks.csv"
HANDOFF_M4_ISSUES_CSV = ROOT / "reports/report3/handoff_member4_missing_and_limited_tracks.csv"
CHORUS_PROCESSED_CSV = ROOT / "data/processed/chorus_repetition_results.csv"

MARKETS_ORDER = ["VN", "US", "KR"]
MARKET_LABELS = {
    "VN": "Vietnam (VN)",
    "US": "United States (US)",
    "KR": "South Korea (KR)",
}
MARKET_COLORS = {
    "VN": "#D95F02",
    "US": "#1B9E77",
    "KR": "#7570B3",
}

NUMERIC_FEATURES = [
    "word_count",
    "unique_word_count",
    "lexical_diversity",
    "repetition_ratio",
    "avg_word_length",
    "line_count",
    "chorus_repetition_count",
    "punctuation_ratio",
    "pronoun_ratio",
]


def load_market_expanded_df(features_df: pd.DataFrame, results_json_path: Path) -> pd.DataFrame:
    """Expand track-level features to (spotify_track_id, market) observations based on sampling_provenance."""
    results_data = json.loads(results_json_path.read_text(encoding="utf-8"))
    records: list[dict[str, Any]] = []

    feat_by_id = features_df.set_index("spotify_track_id").to_dict(orient="index")
    for tid, f_row in feat_by_id.items():
        item = results_data.get(tid, {})
        prov = item.get("sampling_provenance") or []
        markets = sorted({
            str(p.get("market")).strip()
            for p in prov
            if isinstance(p, dict) and p.get("market")
        })
        if not markets and f_row.get("market"):
            markets = [m.strip() for m in str(f_row["market"]).split(";") if m.strip()]
        for mkt in markets:
            rec = {"spotify_track_id": tid, "sampled_market": mkt, **f_row}
            records.append(rec)

    return pd.DataFrame(records)


def compute_numeric_stats(
    series: pd.Series, total_n: int, group_type: str, group_val: str, feature_name: str
) -> dict[str, Any]:
    valid = pd.to_numeric(series, errors="coerce").dropna()
    valid_n = int(len(valid))
    missing_n = int(total_n - valid_n)
    missing_pct = round((missing_n / total_n * 100.0), 2) if total_n > 0 else 0.0

    if valid_n == 0:
        return {
            "group_type": group_type,
            "group_value": group_val,
            "feature": feature_name,
            "total_n": total_n,
            "valid_n": 0,
            "missing_n": missing_n,
            "missing_pct": missing_pct,
            "mean": "",
            "std": "",
            "min": "",
            "q25": "",
            "median": "",
            "q75": "",
            "iqr": "",
            "max": "",
            "true_count": "",
            "false_count": "",
            "true_ratio": "",
        }

    q25 = float(np.percentile(valid, 25))
    q50 = float(np.percentile(valid, 50))
    q75 = float(np.percentile(valid, 75))
    return {
        "group_type": group_type,
        "group_value": group_val,
        "feature": feature_name,
        "total_n": total_n,
        "valid_n": valid_n,
        "missing_n": missing_n,
        "missing_pct": missing_pct,
        "mean": round(float(valid.mean()), 6),
        "std": round(float(valid.std(ddof=1)) if valid_n > 1 else 0.0, 6),
        "min": round(float(valid.min()), 6),
        "q25": round(q25, 6),
        "median": round(q50, 6),
        "q75": round(q75, 6),
        "iqr": round(q75 - q25, 6),
        "max": round(float(valid.max()), 6),
        "true_count": "",
        "false_count": "",
        "true_ratio": "",
    }


def compute_boolean_stats(
    series: pd.Series, total_n: int, group_type: str, group_val: str, feature_name: str
) -> dict[str, Any]:
    cleaned = series.dropna().astype(str).str.strip().str.lower()
    cleaned = cleaned[cleaned.isin(["true", "false"])]
    valid_n = int(len(cleaned))
    missing_n = int(total_n - valid_n)
    missing_pct = round((missing_n / total_n * 100.0), 2) if total_n > 0 else 0.0
    true_cnt = int((cleaned == "true").sum())
    false_cnt = int((cleaned == "false").sum())
    true_ratio = round(true_cnt / valid_n, 6) if valid_n > 0 else ""

    return {
        "group_type": group_type,
        "group_value": group_val,
        "feature": feature_name,
        "total_n": total_n,
        "valid_n": valid_n,
        "missing_n": missing_n,
        "missing_pct": missing_pct,
        "mean": true_ratio,
        "std": "",
        "min": "",
        "q25": "",
        "median": "",
        "q75": "",
        "iqr": "",
        "max": "",
        "true_count": true_cnt if valid_n > 0 else "",
        "false_count": false_cnt if valid_n > 0 else "",
        "true_ratio": true_ratio,
    }


def generate_statistics_csv(features_df: pd.DataFrame, market_df: pd.DataFrame) -> pd.DataFrame:
    stat_rows: list[dict[str, Any]] = []

    # 1. Overall unique tracks (N = 1,912)
    total_unique = len(features_df)
    for feat in NUMERIC_FEATURES:
        stat_rows.append(
            compute_numeric_stats(features_df[feat], total_unique, "overall_unique", "ALL_TRACKS", feat)
        )
    stat_rows.append(
        compute_boolean_stats(
            features_df["title_word_in_lyrics"], total_unique, "overall_unique", "ALL_TRACKS", "title_word_in_lyrics"
        )
    )

    # 2. By Market (VN, US, KR)
    for mkt in MARKETS_ORDER:
        sub = market_df[market_df["sampled_market"] == mkt]
        n_mkt = len(sub)
        for feat in NUMERIC_FEATURES:
            stat_rows.append(compute_numeric_stats(sub[feat], n_mkt, "market", mkt, feat))
        stat_rows.append(
            compute_boolean_stats(sub["title_word_in_lyrics"], n_mkt, "market", mkt, "title_word_in_lyrics")
        )

    # 3. By Primary Language (en, vi, ko, mixed, es, other)
    for lang in ["en", "vi", "ko", "mixed", "es"]:
        sub = features_df[features_df["language"] == lang]
        n_lang = len(sub)
        for feat in NUMERIC_FEATURES:
            stat_rows.append(compute_numeric_stats(sub[feat], n_lang, "language", lang, feat))
        stat_rows.append(
            compute_boolean_stats(sub["title_word_in_lyrics"], n_lang, "language", lang, "title_word_in_lyrics")
        )

    stats_df = pd.DataFrame(stat_rows)
    STATS_CSV.parent.mkdir(parents=True, exist_ok=True)
    stats_df.to_csv(STATS_CSV, index=False, encoding="utf-8-sig")
    return stats_df


def plot_boxplot_feature(
    market_df: pd.DataFrame,
    feature: str,
    title: str,
    ylabel: str,
    chart_csv_name: str,
    fig_png_name: str,
    caveat_note: str,
) -> None:
    sub = market_df[
        ["spotify_track_id", "title", "artist", "sampled_market", "language", "extraction_status", feature]
    ].copy()
    sub[feature] = pd.to_numeric(sub[feature], errors="coerce")

    # Save chart source data (including only valid non-null observations used in plot,
    # while recording sample sizes in header/summary)
    valid_sub = sub.dropna(subset=[feature]).copy()
    valid_sub = valid_sub.rename(columns={"sampled_market": "market"})
    CHART_DATA_DIR.mkdir(parents=True, exist_ok=True)
    valid_sub.to_csv(CHART_DATA_DIR / chart_csv_name, index=False, encoding="utf-8-sig")

    fig, ax = plt.subplots(figsize=(9.5, 6.2), dpi=300)
    sns.set_theme(style="whitegrid")

    x_labels = []
    data_groups = []
    colors = []
    medians = []
    means = []

    for mkt in MARKETS_ORDER:
        mkt_total = int((sub["sampled_market"] == mkt).sum())
        vals = valid_sub.loc[valid_sub["market"] == mkt, feature].values
        valid_n = len(vals)
        x_labels.append(f"{MARKET_LABELS[mkt]}\n(valid n = {valid_n} / N = {mkt_total})")
        data_groups.append(vals)
        colors.append(MARKET_COLORS[mkt])
        medians.append(float(np.median(vals)) if valid_n else np.nan)
        means.append(float(np.mean(vals)) if valid_n else np.nan)

    bp = ax.boxplot(
        data_groups,
        tick_labels=x_labels,
        patch_artist=True,
        widths=0.48,
        showmeans=True,
        meanprops={"marker": "D", "markerfacecolor": "white", "markeredgecolor": "#111111", "markersize": 6},
        medianprops={"color": "#111111", "linewidth": 2.0},
        flierprops={"marker": "o", "markersize": 3.5, "alpha": 0.35, "markeredgecolor": "#444444"},
    )

    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.78)
        patch.set_edgecolor("#222222")
        patch.set_linewidth(1.2)

    # Annotate median and mean above/inside each group
    y_min, y_max = ax.get_ylim()
    y_span = y_max - y_min if y_max > y_min else 1.0
    ax.set_ylim(y_min, y_max + 0.12 * y_span)

    for idx, (med, mn) in enumerate(zip(medians, means), start=1):
        if not np.isnan(med):
            fmt = "{:.1f}" if feature in ("word_count", "line_count") else (
                "{:.2f}" if feature in ("avg_word_length", "chorus_repetition_count") else "{:.3f}"
            )
            ax.text(
                idx,
                y_max + 0.03 * y_span,
                f"Median: {fmt.format(med)} | Mean: {fmt.format(mn)}",
                ha="center",
                va="bottom",
                fontsize=9,
                fontweight="bold",
                color="#222222",
                bbox={"boxstyle": "round,pad=0.25", "facecolor": "#F8F9FA", "edgecolor": "#CCCCCC"},
            )

    ax.set_title(title, fontsize=13, fontweight="bold", pad=14)
    ax.set_xlabel("Sampled Chart Market (with valid observation counts)", fontsize=11, fontweight="bold", labelpad=8)
    ax.set_ylabel(ylabel, fontsize=11, fontweight="bold")
    ax.grid(axis="y", linestyle="--", alpha=0.6)

    fig.text(
        0.5,
        0.01,
        caveat_note,
        ha="center",
        va="bottom",
        fontsize=8.5,
        style="italic",
        color="#444444",
    )
    plt.tight_layout(rect=[0.0, 0.04, 1.0, 1.0])
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURES_DIR / fig_png_name, dpi=300)
    plt.close(fig)


def plot_pronoun_ratio_barchart(market_df: pd.DataFrame) -> None:
    sub = market_df[
        ["spotify_track_id", "title", "artist", "sampled_market", "language", "extraction_status", "pronoun_ratio"]
    ].copy()
    sub["pronoun_ratio"] = pd.to_numeric(sub["pronoun_ratio"], errors="coerce")

    valid_sub = sub.dropna(subset=["pronoun_ratio"]).copy()
    valid_sub = valid_sub.rename(columns={"sampled_market": "market"})
    valid_sub.to_csv(CHART_DATA_DIR / "07_pronoun_ratio_by_market.csv", index=False, encoding="utf-8-sig")

    summary_rows = []
    for mkt in MARKETS_ORDER:
        mkt_total = int((sub["sampled_market"] == mkt).sum())
        vals = valid_sub.loc[valid_sub["market"] == mkt, "pronoun_ratio"].values
        valid_n = len(vals)
        mean_val = float(np.mean(vals))
        std_val = float(np.std(vals, ddof=1))
        ci95 = 1.96 * std_val / math.sqrt(valid_n) if valid_n > 1 else 0.0
        med_val = float(np.median(vals))
        summary_rows.append({
            "market": mkt,
            "label": f"{MARKET_LABELS[mkt]}\n(valid n = {valid_n} / N = {mkt_total})",
            "total_n": mkt_total,
            "valid_n": valid_n,
            "missing_n": mkt_total - valid_n,
            "mean": mean_val,
            "median": med_val,
            "std": std_val,
            "ci95": ci95,
        })

    sum_df = pd.DataFrame(summary_rows)
    fig, ax = plt.subplots(figsize=(9.5, 6.2), dpi=300)

    x = np.arange(len(MARKETS_ORDER))
    width = 0.36

    bars_mean = ax.bar(
        x - width / 2,
        sum_df["mean"],
        width,
        yerr=sum_df["ci95"],
        capsize=5,
        label="Mean Pronoun Ratio (±95% CI)",
        color=[MARKET_COLORS[m] for m in MARKETS_ORDER],
        alpha=0.85,
        edgecolor="#222222",
    )
    bars_med = ax.bar(
        x + width / 2,
        sum_df["median"],
        width,
        label="Median Pronoun Ratio",
        color=[MARKET_COLORS[m] for m in MARKETS_ORDER],
        alpha=0.45,
        edgecolor="#222222",
        hatch="//",
    )

    for bar in bars_mean:
        h = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            h + 0.006,
            f"{h:.3f}\n({h * 100:.1f}%)",
            ha="center",
            va="bottom",
            fontsize=9,
            fontweight="bold",
        )
    for bar in bars_med:
        h = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            h + 0.004,
            f"{h:.3f}\n({h * 100:.1f}%)",
            ha="center",
            va="bottom",
            fontsize=8.5,
        )

    ax.set_xticks(x)
    ax.set_xticklabels(sum_df["label"], fontsize=10)
    ax.set_ylim(0, max(sum_df["mean"] + sum_df["ci95"]) * 1.28)
    ax.set_title("Chart 7: Comparison of Pronoun Ratio Across Sampled Markets (VN, US, KR)", fontsize=13, fontweight="bold", pad=14)
    ax.set_xlabel("Sampled Chart Market (NULL excluded; mixed/unsupported languages not coerced to 0)", fontsize=10.5, fontweight="bold", labelpad=8)
    ax.set_ylabel("Pronoun Ratio (Matched Pronoun Tokens / Word Count)", fontsize=11, fontweight="bold")
    ax.legend(loc="upper right", frameon=True)
    ax.grid(axis="y", linestyle="--", alpha=0.6)

    fig.text(
        0.5,
        0.01,
        "Note: Pronoun ratio is measured for single-language tracks (en, vi, ko, es); mixed-language tracks remain NULL. Korean is pro-drop with agglutinative eojeols.",
        ha="center",
        va="bottom",
        fontsize=8.5,
        style="italic",
        color="#444444",
    )
    plt.tight_layout(rect=[0.0, 0.04, 1.0, 1.0])
    fig.savefig(FIGURES_DIR / "07_barchart_pronoun_ratio_by_market.png", dpi=300)
    plt.close(fig)


def plot_title_word_in_lyrics_chart(market_df: pd.DataFrame) -> None:
    sub = market_df[
        ["spotify_track_id", "title", "artist", "sampled_market", "language", "extraction_status", "title_word_in_lyrics"]
    ].copy()
    sub["clean_bool"] = sub["title_word_in_lyrics"].astype(str).str.strip().str.lower()
    valid_sub = sub[sub["clean_bool"].isin(["true", "false"])].copy()
    valid_sub["title_word_in_lyrics_bool"] = valid_sub["clean_bool"] == "true"
    valid_sub = valid_sub.rename(columns={"sampled_market": "market"})
    valid_sub.drop(columns=["clean_bool"]).to_csv(
        CHART_DATA_DIR / "08_title_word_in_lyrics_by_market.csv", index=False, encoding="utf-8-sig"
    )

    rows = []
    for mkt in MARKETS_ORDER:
        mkt_total = int((sub["sampled_market"] == mkt).sum())
        m_sub = valid_sub[valid_sub["market"] == mkt]
        valid_n = len(m_sub)
        true_n = int(m_sub["title_word_in_lyrics_bool"].sum())
        false_n = valid_n - true_n
        true_pct = (true_n / valid_n * 100.0) if valid_n else 0.0
        false_pct = (false_n / valid_n * 100.0) if valid_n else 0.0
        rows.append({
            "market": mkt,
            "label": f"{MARKET_LABELS[mkt]}\n(valid n = {valid_n} / N = {mkt_total})",
            "true_n": true_n,
            "false_n": false_n,
            "null_n": mkt_total - valid_n,
            "true_pct": true_pct,
            "false_pct": false_pct,
        })

    df_plot = pd.DataFrame(rows)
    fig, ax = plt.subplots(figsize=(9.5, 6.2), dpi=300)

    x = np.arange(len(MARKETS_ORDER))
    width = 0.38

    b_true = ax.bar(
        x - width / 2,
        df_plot["true_pct"],
        width,
        label="Title Content Word Appears in Lyrics (True)",
        color="#2B8CBE",
        edgecolor="#222222",
    )
    b_false = ax.bar(
        x + width / 2,
        df_plot["false_pct"],
        width,
        label="Title Content Word Absent from Lyrics (False)",
        color="#E34A33",
        alpha=0.85,
        edgecolor="#222222",
    )

    for idx, bar in enumerate(b_true):
        h = bar.get_height()
        cnt = df_plot.loc[idx, "true_n"]
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            h + 1.2,
            f"{h:.1f}%\n(n={cnt})",
            ha="center",
            va="bottom",
            fontsize=9,
            fontweight="bold",
        )
    for idx, bar in enumerate(b_false):
        h = bar.get_height()
        cnt = df_plot.loc[idx, "false_n"]
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            h + 1.2,
            f"{h:.1f}%\n(n={cnt})",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    ax.set_xticks(x)
    ax.set_xticklabels(df_plot["label"], fontsize=10)
    ax.set_ylim(0, 108)
    ax.set_title("Chart 8: Title Content Word Presence in Lyrics Across Markets (VN, US, KR)", fontsize=13, fontweight="bold", pad=14)
    ax.set_xlabel("Sampled Chart Market (NULL excluded when title has only stopwords/numbers or unsupported language)", fontsize=10, fontweight="bold", labelpad=8)
    ax.set_ylabel("Percentage of Valid Tracks (%)", fontsize=11, fontweight="bold")
    ax.legend(loc="upper right", frameon=True)
    ax.grid(axis="y", linestyle="--", alpha=0.6)

    fig.text(
        0.5,
        0.01,
        "Note: Korean/Vietnamese songs often use English or metaphorical titles that do not appear verbatim in Hangul/Vietnamese lyric bodies.",
        ha="center",
        va="bottom",
        fontsize=8.5,
        style="italic",
        color="#444444",
    )
    plt.tight_layout(rect=[0.0, 0.04, 1.0, 1.0])
    fig.savefig(FIGURES_DIR / "08_barchart_title_word_in_lyrics_by_market.png", dpi=300)
    plt.close(fig)


def plot_language_distribution_by_market(market_df: pd.DataFrame) -> None:
    sub = market_df.copy()
    sub["lang_group"] = sub["language"].apply(
        lambda l: l if l in ("en", "vi", "ko", "mixed", "es") else "other/unknown"
    )
    ctab = pd.crosstab(sub["sampled_market"], sub["lang_group"]).reindex(MARKETS_ORDER)
    cols_order = [c for c in ["en", "vi", "ko", "mixed", "es", "other/unknown"] if c in ctab.columns]
    ctab = ctab[cols_order]
    ctab_pct = ctab.div(ctab.sum(axis=1), axis=0) * 100.0

    out_records = []
    for mkt in MARKETS_ORDER:
        for col in cols_order:
            out_records.append({
                "market": mkt,
                "language_group": col,
                "track_count": int(ctab.loc[mkt, col]),
                "percentage": round(float(ctab_pct.loc[mkt, col]), 2),
            })
    pd.DataFrame(out_records).to_csv(
        CHART_DATA_DIR / "09_language_distribution_by_market.csv", index=False, encoding="utf-8-sig"
    )

    fig, ax = plt.subplots(figsize=(9.5, 6.0), dpi=300)
    palette = {
        "en": "#1B9E77",
        "vi": "#D95F02",
        "ko": "#7570B3",
        "mixed": "#E7298A",
        "es": "#66A61E",
        "other/unknown": "#999999",
    }
    bottom = np.zeros(len(MARKETS_ORDER))
    x = np.arange(len(MARKETS_ORDER))

    for col in cols_order:
        vals = ctab_pct[col].values
        cnts = ctab[col].values
        bars = ax.bar(
            x,
            vals,
            bottom=bottom,
            width=0.52,
            label=col,
            color=palette.get(col, "#CCCCCC"),
            edgecolor="#222222",
        )
        for i, (v, c) in enumerate(zip(vals, cnts)):
            if v >= 5.0:
                ax.text(
                    x[i],
                    bottom[i] + v / 2,
                    f"{col}: {v:.1f}% (n={c})",
                    ha="center",
                    va="center",
                    fontsize=8.5,
                    fontweight="bold",
                    color="white" if col in ("en", "vi", "ko", "mixed") else "#111111",
                )
        bottom += vals

    x_labels = [
        f"{MARKET_LABELS[m]}\n(Total N = {int(ctab.loc[m].sum())})"
        for m in MARKETS_ORDER
    ]
    ax.set_xticks(x)
    ax.set_xticklabels(x_labels, fontsize=10)
    ax.set_ylim(0, 105)
    ax.set_title("Supplementary Chart 9: Lyric Language Distribution Within Each Sampled Market", fontsize=13, fontweight="bold", pad=14)
    ax.set_xlabel("Sampled Chart Market (Market != Lyric Language)", fontsize=11, fontweight="bold", labelpad=8)
    ax.set_ylabel("Share of Sampled Market Tracks (%)", fontsize=11, fontweight="bold")
    ax.legend(title="Detected Language", bbox_to_anchor=(1.01, 1), loc="upper left", frameon=True)
    ax.grid(axis="y", linestyle="--", alpha=0.6)
    plt.tight_layout()
    fig.savefig(FIGURES_DIR / "09_stacked_language_by_market.png", dpi=300)
    plt.close(fig)


def generate_handoff_files(features_df: pd.DataFrame, market_df: pd.DataFrame) -> None:
    # 1. Member 1 handoff: (spotify_track_id, market) join table (2,098 rows, retaining all tracks including not_found)
    m1_df = market_df.copy()
    m1_df = m1_df.rename(columns={"sampled_market": "market", "market": "all_sampled_markets"})
    m1_df["has_valid_lyrics"] = m1_df["lyrics_status"] == "success"
    m1_df["is_full_feature_set"] = m1_df["extraction_status"] == "success"
    m1_cols = [
        "spotify_track_id",
        "market",
        "all_sampled_markets",
        "title",
        "artist",
        "language",
        "source",
        "lyrics_status",
        "extraction_status",
        "has_valid_lyrics",
        "is_full_feature_set",
        *NUMERIC_FEATURES[:7],
        "title_word_in_lyrics",
        "punctuation_ratio",
        "pronoun_ratio",
        "rules_version",
    ]
    m1_df = m1_df[m1_cols].sort_values(["market", "spotify_track_id"])
    m1_df.to_csv(HANDOFF_M1_CSV, index=False, encoding="utf-8-sig")

    # 2. Member 4 handoffs: valid tracks vs missing/language-limited tracks
    valid_df = features_df[features_df["lyrics_status"] == "success"].copy()
    valid_df.to_csv(HANDOFF_M4_VALID_CSV, index=False, encoding="utf-8-sig")

    issues_df = features_df[features_df["extraction_status"] != "success"].copy()
    issues_df.to_csv(HANDOFF_M4_ISSUES_CSV, index=False, encoding="utf-8-sig")

    # 3. Synchronize data/processed/chorus_repetition_results.csv (1,909 tracks with lyrics)
    chorus_df = valid_df[["spotify_track_id", "chorus_repetition_count", "lyrics_status"]].copy()
    chorus_df["error"] = chorus_df["chorus_repetition_count"].apply(
        lambda v: "chorus_unidentified_no_explicit_tag_or_repeated_block" if pd.isna(v) or str(v).strip() == "" else ""
    )
    chorus_df.to_csv(CHORUS_PROCESSED_CSV, index=False, encoding="utf-8-sig")


def main() -> int:
    features_df = pd.read_csv(FEATURES_CSV, dtype=str, keep_default_na=False)
    market_df = load_market_expanded_df(features_df, RESULTS_JSON)

    stats_df = generate_statistics_csv(features_df, market_df)
    print(f"Saved statistics CSV: {STATS_CSV} ({len(stats_df)} rows)")

    caveat_token = (
        "Methodological Note: Market != Language. Tokenization differs across English (words), "
        "Vietnamese (whitespace syllables), and Korean (whitespace eojeols)."
    )

    plot_boxplot_feature(
        market_df,
        "word_count",
        "Chart 1: Distribution of Lyric Word Count by Sampled Market (VN, US, KR)",
        "Word Count (tokens per song)",
        "01_word_count_by_market.csv",
        "01_boxplot_word_count_by_market.png",
        caveat_token,
    )
    plot_boxplot_feature(
        market_df,
        "lexical_diversity",
        "Chart 2: Distribution of Lexical Diversity (TTR) by Sampled Market (VN, US, KR)",
        "Lexical Diversity (Unique Tokens / Total Tokens)",
        "02_lexical_diversity_by_market.csv",
        "02_boxplot_lexical_diversity_by_market.png",
        caveat_token,
    )
    plot_boxplot_feature(
        market_df,
        "repetition_ratio",
        "Chart 3: Distribution of Token Repetition Ratio by Sampled Market (VN, US, KR)",
        "Repetition Ratio ((Word Count - Unique Words) / Word Count)",
        "03_repetition_ratio_by_market.csv",
        "03_boxplot_repetition_ratio_by_market.png",
        caveat_token,
    )
    plot_boxplot_feature(
        market_df,
        "avg_word_length",
        "Chart 4: Distribution of Average Token Length by Sampled Market (VN, US, KR)",
        "Average Word Length (Unicode characters per token)",
        "04_avg_word_length_by_market.csv",
        "04_boxplot_avg_word_length_by_market.png",
        "Note: Korean Hangul characters represent whole syllables (short char length); Vietnamese tokens are single syllables; English tokens are whole words.",
    )
    plot_boxplot_feature(
        market_df,
        "line_count",
        "Chart 5: Distribution of Non-Empty Lyric Line Count by Sampled Market (VN, US, KR)",
        "Line Count (non-empty lines per song)",
        "05_line_count_by_market.csv",
        "05_boxplot_line_count_by_market.png",
        "Note: Line count depends on lyric transcription formatting (Genius/LRCLIB line breaks) as well as song length.",
    )
    plot_boxplot_feature(
        market_df,
        "chorus_repetition_count",
        "Chart 6: Distribution of Chorus Repetition Count by Sampled Market (VN, US, KR)",
        "Additional Chorus Occurrences (count)",
        "06_chorus_repetition_count_by_market.csv",
        "06_boxplot_chorus_repetition_count_by_market.png",
        "Note: Unidentified choruses are kept NULL (excluded here, never coerced to 0). Value 0 occurs only with explicit [Chorus] tag and 0 repeats.",
    )
    plot_pronoun_ratio_barchart(market_df)
    plot_title_word_in_lyrics_chart(market_df)
    plot_language_distribution_by_market(market_df)

    generate_handoff_files(features_df, market_df)
    print(f"Saved 9 chart CSVs to: {CHART_DATA_DIR}")
    print(f"Saved 9 PNG figures to: {FIGURES_DIR}")
    print(f"Saved handoff files: {HANDOFF_M1_CSV.name}, {HANDOFF_M4_VALID_CSV.name}, {HANDOFF_M4_ISSUES_CSV.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
