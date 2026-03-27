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


def load_accuracy(path: Path) -> float:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return float(data["accuracy"])


def plot_fixed_chunk_accuracy():
    labels = ["256", "512", "1024"]
    values = [load_accuracy(EVAL_PATHS[label]) for label in labels]

    plt.figure(figsize=(7, 4.5))
    plt.bar(labels, values)
    plt.ylim(0, 1.0)
    plt.xlabel("Chunk Size")
    plt.ylabel("Accuracy")
    plt.title("Accuracy by Fixed Chunk Size")

    for i, v in enumerate(values):
        plt.text(i, v + 0.02, f"{v:.2f}", ha="center")

    out_path = FIG_DIR / "chunk_size_accuracy.png"
    plt.tight_layout()
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out_path}")


def plot_fixed_vs_adaptive():
    labels = ["Fixed-256", "Fixed-512", "Fixed-1024", "Adaptive"]
    values = [
        load_accuracy(EVAL_PATHS["256"]),
        load_accuracy(EVAL_PATHS["512"]),
        load_accuracy(EVAL_PATHS["1024"]),
        load_accuracy(EVAL_PATHS["Adaptive"]),
    ]

    plt.figure(figsize=(8, 4.8))
    plt.bar(labels, values)
    plt.ylim(0, 1.0)
    plt.xlabel("Method")
    plt.ylabel("Accuracy")
    plt.title("Fixed Chunking vs Adaptive Chunking")

    for i, v in enumerate(values):
        plt.text(i, v + 0.02, f"{v:.2f}", ha="center")

    out_path = FIG_DIR / "fixed_vs_adaptive_accuracy.png"
    plt.tight_layout()
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out_path}")


def main():
    plot_fixed_chunk_accuracy()
    plot_fixed_vs_adaptive()


if __name__ == "__main__":
    main()