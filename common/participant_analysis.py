"""Shared supplementary-analysis logic for user-study participant data:
time-on-task vs. support modality, decisiveness (answer/concept-value
changes) vs. support modality, and trust/confidence/interaction/time
correlations. Used by both emails/scripts/supplementary_analysis.py and
cub/scripts/supplementary_analysis.py -- each just supplies its own
CONDITION_ORDER, data directory, and which condition allows concept editing.

Expects a long-format dataframe (one row per participant x stimulus) with
columns: ID, ExperimentalCondition, TimeSpent (minutes), TimesAnswerChanged,
Confidence_Adjusted, and Feature{1..6}_Clicks; and a wide-format dataframe
(one row per participant) with: ID, ExperimentalCondition, TrustIndex,
AverageAccuracy, TotalInteractions.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats

CLICK_COLS = [f"Feature{i}_Clicks" for i in range(1, 7)]

sns.set_theme(style="whitegrid")


def to_num(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


def prep_long_df(long_df: pd.DataFrame, condition_order: list[str]) -> pd.DataFrame:
    long_df = long_df.copy()
    long_df["ExperimentalCondition"] = pd.Categorical(long_df["ExperimentalCondition"], categories=condition_order, ordered=True)
    long_df["total_clicks"] = long_df[CLICK_COLS].apply(to_num).sum(axis=1)
    return long_df


def prep_wide_df(wide_df: pd.DataFrame, condition_order: list[str]) -> pd.DataFrame:
    wide_df = wide_df.copy()
    wide_df["ExperimentalCondition"] = pd.Categorical(wide_df["ExperimentalCondition"], categories=condition_order, ordered=True)
    wide_df["TrustIndex"] = to_num(wide_df["TrustIndex"])
    return wide_df


def kruskal_by_condition(df: pd.DataFrame, value_col: str) -> tuple[float, float]:
    groups = [g[value_col].dropna().values for _, g in df.groupby("ExperimentalCondition", observed=True)]
    return stats.kruskal(*groups)


def boxplot_by_condition(df: pd.DataFrame, value_col: str, title: str, ylabel: str,
                          out_path: Path, condition_order: list[str], dataset_name: str) -> None:
    fig, ax = plt.subplots(figsize=(7, 5))
    sns.boxplot(data=df, x="ExperimentalCondition", y=value_col, order=condition_order, ax=ax)
    sns.stripplot(data=df, x="ExperimentalCondition", y=value_col, order=condition_order,
                  color="black", alpha=0.25, size=3, ax=ax)
    ax.set_title(f"[{dataset_name}] {title}")
    ax.set_xlabel("Support modality")
    ax.set_ylabel(ylabel)
    plt.xticks(rotation=15)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  saved {out_path}")


def analysis_time_vs_support(long_df: pd.DataFrame, wide_df: pd.DataFrame, out_dir: Path,
                              condition_order: list[str], dataset_name: str) -> None:
    print("\n=== A. Time-on-task vs. support modality ===")

    boxplot_by_condition(long_df, "TimeSpent", "Per-stimulus time spent, by support modality",
                          "Time spent (minutes)", out_dir / "time_per_stimulus_by_condition.png",
                          condition_order, dataset_name)
    h, p = kruskal_by_condition(long_df, "TimeSpent")
    print(f"  Kruskal-Wallis (per-stimulus TimeSpent across conditions): H={h:.2f}, p={p:.4g}")
    print(long_df.groupby("ExperimentalCondition", observed=True)["TimeSpent"].agg(["mean", "median", "std", "count"]))

    total_time = long_df.groupby("ID")["TimeSpent"].sum().rename("TotalTimeSpent")
    participant_time = wide_df[["ID", "ExperimentalCondition"]].merge(total_time, on="ID")
    boxplot_by_condition(participant_time, "TotalTimeSpent", "Total time on task per participant, by support modality",
                          "Total time (minutes)", out_dir / "total_time_per_participant_by_condition.png",
                          condition_order, dataset_name)
    h, p = kruskal_by_condition(participant_time, "TotalTimeSpent")
    print(f"  Kruskal-Wallis (per-participant total TimeSpent across conditions): H={h:.2f}, p={p:.4g}")
    print(participant_time.groupby("ExperimentalCondition", observed=True)["TotalTimeSpent"].agg(["mean", "median", "std", "count"]))


def analysis_decisiveness(long_df: pd.DataFrame, out_dir: Path, condition_order: list[str],
                           interactive_condition: str, dataset_name: str) -> None:
    print("\n=== B. Decisiveness vs. support modality ===")

    # TimesAnswerChanged is a heavily zero-inflated count (median 0 in every
    # condition -- most participants never revise their classification), so
    # a boxplot is uninformative (just a flat line at 0). Plot the share of
    # stimuli with >=1 change instead, which is where the real signal is.
    changed_share = (
        long_df.assign(changed=long_df["TimesAnswerChanged"] > 0)
        .groupby("ExperimentalCondition", observed=True)["changed"]
        .mean()
        .reindex(condition_order)
        .reset_index()
    )
    fig, ax = plt.subplots(figsize=(7, 5))
    sns.barplot(data=changed_share, x="ExperimentalCondition", y="changed", order=condition_order, ax=ax)
    ax.set_title(f"[{dataset_name}] Share of stimuli where the classification answer was changed at least once")
    ax.set_xlabel("Support modality")
    ax.set_ylabel("Share of stimuli with >=1 answer change")
    ax.yaxis.set_major_formatter(lambda v, _: f"{v:.0%}")
    plt.xticks(rotation=15)
    fig.tight_layout()
    fig.savefig(out_dir / "answer_changes_by_condition.png", dpi=150)
    plt.close(fig)
    print(f"  saved {out_dir / 'answer_changes_by_condition.png'}")

    h, p = kruskal_by_condition(long_df, "TimesAnswerChanged")
    print(f"  Kruskal-Wallis (TimesAnswerChanged across conditions): H={h:.2f}, p={p:.4g}")
    print(long_df.groupby("ExperimentalCondition", observed=True)["TimesAnswerChanged"].agg(["mean", "median", "std", "count"]))

    # Concept-value editing (clicking a Feature box to flip it) is only
    # possible in the fully-interactive condition -- other conditions don't
    # expose an editable concept UI, so total_clicks is 0 there by design.
    interactive = long_df[long_df["ExperimentalCondition"] == interactive_condition]
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.histplot(interactive["total_clicks"], discrete=True, ax=ax)
    ax.set_title(f"[{dataset_name}] Concept-value edits per stimulus ({interactive_condition} only)")
    ax.set_xlabel("Total concept clicks (sum over 6 features)")
    fig.tight_layout()
    fig.savefig(out_dir / "concept_clicks_interactive_only.png", dpi=150)
    plt.close(fig)
    print(f"  saved {out_dir / 'concept_clicks_interactive_only.png'}")
    print(f"  {interactive_condition}: {(interactive['total_clicks'] > 0).mean():.1%} of stimuli had "
          f"at least one concept edit; mean {interactive['total_clicks'].mean():.2f} edits/stimulus")


def analysis_correlations(long_df: pd.DataFrame, wide_df: pd.DataFrame, out_dir: Path,
                           condition_order: list[str], dataset_name: str) -> None:
    print("\n=== C. Trust / confidence / interaction / time correlations ===")

    per_participant = long_df.groupby("ID").agg(
        MeanConfidenceAdjusted=("Confidence_Adjusted", "mean"),
        MeanTimesAnswerChanged=("TimesAnswerChanged", "mean"),
        TotalTimeSpent=("TimeSpent", "sum"),
        TotalClicks=("total_clicks", "sum"),
    ).reset_index()
    table = wide_df[["ID", "ExperimentalCondition", "TrustIndex", "AverageAccuracy", "TotalInteractions"]].merge(
        per_participant, on="ID"
    )

    corr_cols = ["TrustIndex", "AverageAccuracy", "MeanConfidenceAdjusted", "MeanTimesAnswerChanged",
                 "TotalTimeSpent", "TotalInteractions"]
    corr = table[corr_cols].corr(method="spearman")

    fig, ax = plt.subplots(figsize=(8, 6.5))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="RdBu_r", vmin=-1, vmax=1, ax=ax)
    ax.set_title(f"[{dataset_name}] Spearman correlation (per-participant aggregates)\n"
                  "TrustIndex is NaN where there's no model to rate, pairwise-excluded there", fontsize=10)
    fig.tight_layout()
    fig.savefig(out_dir / "correlation_heatmap.png", dpi=150)
    plt.close(fig)
    print(f"  saved {out_dir / 'correlation_heatmap.png'}")
    print(corr.round(3))

    print("\n  Pairwise Spearman (coefficient, p-value, n):")
    pairs = [
        ("TrustIndex", "MeanConfidenceAdjusted"),
        ("TrustIndex", "MeanTimesAnswerChanged"),
        ("TrustIndex", "TotalTimeSpent"),
        ("MeanConfidenceAdjusted", "MeanTimesAnswerChanged"),
        ("MeanConfidenceAdjusted", "TotalTimeSpent"),
        ("MeanTimesAnswerChanged", "TotalTimeSpent"),
    ]
    for x, y in pairs:
        sub = table[[x, y]].dropna()
        rho, p = stats.spearmanr(sub[x], sub[y])
        print(f"    {x} vs {y}: rho={rho:+.3f}, p={p:.4g}, n={len(sub)}")

    fig, axes = plt.subplots(2, 3, figsize=(16, 9))
    for ax, (x, y) in zip(axes.flat, pairs):
        sns.scatterplot(data=table, x=x, y=y, hue="ExperimentalCondition", hue_order=condition_order,
                         alpha=0.7, ax=ax, legend=(ax is axes.flat[0]))
        sub = table[[x, y]].dropna()
        if len(sub) > 2:
            slope, intercept, *_ = stats.linregress(sub[x], sub[y])
            xs = np.linspace(sub[x].min(), sub[x].max(), 50)
            ax.plot(xs, slope * xs + intercept, color="black", linestyle="--", linewidth=1)
        ax.set_title(f"{x} vs. {y}")
    if axes.flat[0].get_legend() is not None:
        axes.flat[0].legend(fontsize=7, loc="best")
    fig.suptitle(f"[{dataset_name}] Pairwise correlations (per-participant aggregates)")
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(out_dir / "correlation_scatterplots.png", dpi=150)
    plt.close(fig)
    print(f"  saved {out_dir / 'correlation_scatterplots.png'}")


def run_all(long_df: pd.DataFrame, wide_df: pd.DataFrame, out_dir: Path,
            condition_order: list[str], interactive_condition: str, dataset_name: str) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    long_df = prep_long_df(long_df, condition_order)
    wide_df = prep_wide_df(wide_df, condition_order)
    print(f"[{dataset_name}] Loaded long ({len(long_df)} rows) and wide ({len(wide_df)} rows) participant data")

    analysis_time_vs_support(long_df, wide_df, out_dir, condition_order, dataset_name)
    analysis_decisiveness(long_df, out_dir, condition_order, interactive_condition, dataset_name)
    analysis_correlations(long_df, wide_df, out_dir, condition_order, dataset_name)

    expected = [
        "time_per_stimulus_by_condition.png",
        "total_time_per_participant_by_condition.png",
        "answer_changes_by_condition.png",
        "concept_clicks_interactive_only.png",
        "correlation_heatmap.png",
        "correlation_scatterplots.png",
    ]
    missing = [f for f in expected if not (out_dir / f).exists()]
    if missing:
        raise RuntimeError(f"[{dataset_name}] missing expected figures after generation: {missing}")
    print(f"[{dataset_name}] All {len(expected)} expected figures present in {out_dir}")
