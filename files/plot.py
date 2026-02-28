import pandas as pd
import matplotlib.pyplot as plt

# ---- load ----
df = pd.read_csv("xauusd_test_pruned.csv").dropna()

target = df.columns[1]  # 2nd column
num = df.select_dtypes(include="number")

# keep only numeric columns that exist
cols = [c for c in num.columns if c != target]
if target not in num.columns:
    raise ValueError(f"Target (2nd column) '{target}' is not numeric. Encode it or choose another target.")

# optional: reduce clutter
plot_df = df.sample(min(len(df), 3000), random_state=42)

# ---- plot grid: each feature vs target ----
features = [c for c in cols if c in plot_df.columns]
n = len(features)
if n == 0:
    raise ValueError("No numeric features to plot against the target.")

grid_cols = 4
grid_rows = (n + grid_cols - 1) // grid_cols

plt.figure(figsize=(grid_cols * 4, grid_rows * 3))

for i, f in enumerate(features, 1):
    ax = plt.subplot(grid_rows, grid_cols, i)
    ax.scatter(plot_df[f], plot_df[target], s=6, alpha=0.5)
    ax.set_title(f)
    ax.set_xlabel("")
    ax.set_ylabel(target)

plt.suptitle(f"Each feature vs target: {target}", y=1.02)
plt.tight_layout()
plt.show()