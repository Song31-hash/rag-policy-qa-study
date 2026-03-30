import os
import pandas as pd
import matplotlib.pyplot as plt


SUMMARY_PATH = os.path.join("results", "summary.csv")
FIG_DIR = "figures"


def load_summary(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    required_cols = {"experiment", "overlap", "top_k", "accuracy"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns in summary.csv: {missing}")

    df["overlap"] = df["overlap"].astype(int)
    df["top_k"] = df["top_k"].astype(int)
    df["accuracy"] = df["accuracy"].astype(float)
    return df


def plot_heatmap(df: pd.DataFrame, save_path: str) -> None:
    pivot = df.pivot(index="overlap", columns="top_k", values="accuracy")
    pivot = pivot.sort_index().sort_index(axis=1)

    fig, ax = plt.subplots(figsize=(6, 4.5))
    im = ax.imshow(pivot.values)

    ax.set_title("Accuracy Heatmap (Overlap vs Top-k)")
    ax.set_xlabel("Top-k")
    ax.set_ylabel("Overlap")

    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels(pivot.columns)
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels(pivot.index)

    for i in range(len(pivot.index)):
        for j in range(len(pivot.columns)):
            val = pivot.iloc[i, j]
            ax.text(j, i, f"{val:.2f}", ha="center", va="center", color="white")

    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    fig.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_topk(df: pd.DataFrame, save_path: str, overlap_value: int = 64) -> None:
    sub = df[df["overlap"] == overlap_value].sort_values("top_k")
    if sub.empty:
        raise ValueError(f"No rows found for overlap={overlap_value}")

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(sub["top_k"], sub["accuracy"], marker="o")
    ax.set_title(f"Top-k vs Accuracy (Overlap={overlap_value})")
    ax.set_xlabel("Top-k")
    ax.set_ylabel("Accuracy")
    ax.set_xticks(sub["top_k"].tolist())

    for x, y in zip(sub["top_k"], sub["accuracy"]):
        ax.text(x, y + 0.005, f"{y:.2f}", ha="center")

    fig.tight_layout()
    fig.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_overlap(df: pd.DataFrame, save_path: str, topk_value: int = 5) -> None:
    sub = df[df["top_k"] == topk_value].sort_values("overlap")
    if sub.empty:
        raise ValueError(f"No rows found for top_k={topk_value}")

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(sub["overlap"], sub["accuracy"], marker="o")
    ax.set_title(f"Overlap vs Accuracy (Top-k={topk_value})")
    ax.set_xlabel("Overlap")
    ax.set_ylabel("Accuracy")
    ax.set_xticks(sub["overlap"].tolist())

    for x, y in zip(sub["overlap"], sub["accuracy"]):
        ax.text(x, y + 0.005, f"{y:.2f}", ha="center")

    fig.tight_layout()
    fig.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    os.makedirs(FIG_DIR, exist_ok=True)

    df = load_summary(SUMMARY_PATH)

    plot_heatmap(df, os.path.join(FIG_DIR, "heatmap.png"))
    plot_topk(df, os.path.join(FIG_DIR, "topk_plot.png"), overlap_value=64)
    plot_overlap(df, os.path.join(FIG_DIR, "overlap_plot.png"), topk_value=5)

    print("Saved:")
    print("-", os.path.join(FIG_DIR, "heatmap.png"))
    print("-", os.path.join(FIG_DIR, "topk_plot.png"))
    print("-", os.path.join(FIG_DIR, "overlap_plot.png"))


if __name__ == "__main__":
    main()