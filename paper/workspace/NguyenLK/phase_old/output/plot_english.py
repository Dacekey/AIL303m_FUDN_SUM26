import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

data = {
    "Method": [
        "Raw (gray)", "HOG (gray)+HSV", "HOG (gray)", "HOG (YUV)", 
        "HOG (YUV)+HSV", "HOG (CLAHE)", "HOG (CLAHE)+HSV", 
        "HSV", "HOG (Hue)+HSV", "HOG (Hue)"
    ],
    "n_features": [4096, 2276, 1764, 5292, 5804, 1764, 2276, 512, 2276, 1764],
    "train_seconds": [59.9, 40.2, 29.6, 135.8, 142.9, 29.7, 41.6, 3.2, 68.4, 48.2],
    "val_accuracy": [0.943, 0.955, 0.953, 0.955, 0.955, 0.942, 0.943, 0.908, 0.880, 0.766],
    "val_macro_f1": [0.934, 0.933, 0.929, 0.927, 0.921, 0.921, 0.914, 0.906, 0.799, 0.626]
}
df = pd.DataFrame(data)
df = df.sort_values(by="val_accuracy", ascending=False)

fig, ax1 = plt.subplots(figsize=(15, 8))

x = np.arange(len(df["Method"]))
width = 0.35

bars1 = ax1.bar(x - width/2, df["val_accuracy"], width, label='Val Accuracy', color='skyblue')
bars2 = ax1.bar(x + width/2, df["val_macro_f1"], width, label='Val Macro F1', color='lightgreen')

ax1.set_xlabel('Feature Extraction Method', fontsize=12, fontweight='bold')
ax1.set_ylabel('Score (Accuracy / F1)', fontsize=12, fontweight='bold')
ax1.set_xticks(x)
ax1.set_xticklabels(df["Method"], rotation=45, ha='right')
ax1.set_ylim(0.5, 1.1)  # Increased slightly to fit annotations

# Annotate values on the bars
for bar in bars1:
    yval = bar.get_height()
    ax1.text(bar.get_x() + bar.get_width()/2.0, yval + 0.01, f'{yval:.3f}', ha='center', va='bottom', fontsize=9, rotation=90)

for bar in bars2:
    yval = bar.get_height()
    ax1.text(bar.get_x() + bar.get_width()/2.0, yval + 0.01, f'{yval:.3f}', ha='center', va='bottom', fontsize=9, rotation=90)


ax2 = ax1.twinx()
line = ax2.plot(x, df["train_seconds"], color='tomato', marker='o', linewidth=2, markersize=8, label='Train Time (s)')
ax2.set_ylabel('Train Time (seconds)', fontsize=12, fontweight='bold', color='tomato')
ax2.tick_params(axis='y', labelcolor='tomato')
ax2.set_ylim(0, max(df["train_seconds"]) * 1.2) # Give some space for feature annotations

# Annotate n_features
for i, txt in enumerate(df["n_features"]):
    ax2.annotate(f"{txt} dims", (x[i], df["train_seconds"].iloc[i]), textcoords="offset points", xytext=(0,10), ha='center', fontsize=9, color='darkred', weight='bold')

lines_1, labels_1 = ax1.get_legend_handles_labels()
lines_2, labels_2 = ax2.get_legend_handles_labels()
ax1.legend(lines_1 + lines_2, labels_1 + labels_2, loc='upper right')

plt.title('Correlation between Accuracy, Time & Feature Dimensions (Phase 1)', fontsize=14, fontweight='bold')
plt.grid(axis='y', linestyle='--', alpha=0.6)
plt.tight_layout()

plt.savefig('/home/dacekey/AIL303_SUM26/paper/workspace/NguyenLK/phase_old/output/phase1_results_visualization_english.png', dpi=300)
print("Plot saved successfully.")
