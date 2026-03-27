import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os

# ----------------------
# CSV 로드
# ----------------------
df = pd.read_csv("results/tables/summary.csv")

os.makedirs("paper/figures", exist_ok=True)

# ----------------------
# 1. Heatmap
# ----------------------
pivot = df.pivot(index="overlap", columns="top_k", values="accuracy")

plt.figure(figsize=(6,4))
sns.heatmap(pivot, annot=True, fmt=".2f", cmap="Blues")
plt.title("Accuracy Heatmap (Overlap vs Top-k)")
plt.savefig("paper/figures/heatmap.png")
plt.close()

# ----------------------
# 2. Top-k 그래프 (overlap=64 기준)
# ----------------------
df64 = df[df["overlap"] == 64]

plt.figure()
plt.plot(df64["top_k"], df64["accuracy"], marker="o")
plt.xlabel("Top-k")
plt.ylabel("Accuracy")
plt.title("Top-k vs Accuracy (Overlap=64)")
plt.savefig("paper/figures/topk_plot.png")
plt.close()

# ----------------------
# 3. Overlap 그래프 (top-k=5 기준)
# ----------------------
df5 = df[df["top_k"] == 5]

plt.figure()
plt.plot(df5["overlap"], df5["accuracy"], marker="o")
plt.xlabel("Overlap")
plt.ylabel("Accuracy")
plt.title("Overlap vs Accuracy (Top-k=5)")
plt.savefig("paper/figures/overlap_plot.png")
plt.close()

print("All graphs saved to paper/figures/")