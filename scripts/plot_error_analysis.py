from pathlib import Path
import json
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parent.parent
FIG_DIR = PROJECT_ROOT / "paper" / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

EVAL_PATHS = {
    "256": PROJECT_ROOT / "experiments" / "chunk_size" / "chunk256" / "evaluation.json",
    "512": PROJECT_ROOT / "experiments" / "chunk_size" / "chunk512" / "evaluation.json",
    "1024": PROJECT_ROOT / "experiments" / "chunk_size" / "chunk1024" / "evaluation.json",
    "Adaptive": PROJECT_ROOT / "experiments" / "adaptive" / "chunk_router" / "evaluation.json",
}

ERROR_ORDER = [
    "retrieval_failure",
    "partial_retrieval",
    "reasoning_failure",
    "normalization_failure",
]


def load_error_counts(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data.get("error_type_counts", {})


def plot_single_method_error_distribution(method_name: str, eval_path: Path):
    counts = load_error_counts(eval_path)
    labels = ERROR_ORDER
    values = [counts.get(label, 0) for label in labels]

    plt.figure(figsize=(8, 4.5))
    plt.bar(labels, values)
    plt.xlabel("Error Type")
    plt.ylabel("Count")
    plt.title(f"Error Type Distribution - {method_name}")
    plt.xticks(rotation=20)

    for i, v in enumerate(values):
        plt.text(i, v + 0.05, str(v), ha="center")

    out_path = FIG_DIR / f"error_distribution_{method_name.lower()}.png"
    plt.tight_layout()
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out_path}")


def plot_comparison_error_distribution():
    methods = list(EVAL_PATHS.keys())
    error_matrix = {err: [] for err in ERROR_ORDER}

    for method in methods:
        counts = load_error_counts(EVAL_PATHS[method])
        for err in ERROR_ORDER:
            error_matrix[err].append(counts.get(err, 0))

    x = range(len(methods))
    width = 0.2

    plt.figure(figsize=(10, 5.2))
    offsets = [-1.5, -0.5, 0.5, 1.5]

    for offset, err in zip(offsets, ERROR_ORDER):
        positions = [i + offset * width for i in x]
        plt.bar(positions, error_matrix[err], width=width, label=err)

    plt.xticks(list(x), methods)
    plt.xlabel("Method")
    plt.ylabel("Count")
    plt.title("Error Type Comparison Across Methods")
    plt.legend()

    out_path = FIG_DIR / "error_distribution_comparison.png"
    plt.tight_layout()
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out_path}")


def main():
    for method, path in EVAL_PATHS.items():
        plot_single_method_error_distribution(method, path)

    plot_comparison_error_distribution()


if __name__ == "__main__":
    main()