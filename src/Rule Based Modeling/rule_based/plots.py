# ============================================================
# FULL RESULTS DATASET + FOUR FINAL PLOTS
# Updated with latest FINAL 10-model results
# Generates:
#   plot1_accuracy.png
#   plot2_f1_score.png
#   plot3_f1_vs_space.png
#   plot4_f1_vs_time.png
#   all_model_results.csv
# ============================================================

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

sns.set(style="whitegrid", font_scale=1.15)

# ============================================================
# FINAL RESULTS (LATEST)
# ============================================================

data = [
# Transformer, Mode, Accuracy, Subset_Accuracy, F1_Macro, F1_Micro,
# Hamming_Loss, Train_Time_sec, Feature_Space_MB

["roberta","transformer_only",0.794332,0.089506,0.318911,0.405515,0.205668,2.068472,23.552628],
["roberta","embedding",0.775814,0.055556,0.337662,0.418063,0.224186,5.689216,47.110863],
["roberta","combined",0.782828,0.074074,0.331732,0.413636,0.217172,205.100107,71.224087],

["sentence-transformers","transformer_only",0.765713,0.049383,0.237397,0.374532,0.234287,0.815572,8.835831],
["sentence-transformers","embedding",0.738496,0.009259,0.246260,0.364256,0.261504,1.831661,17.654282],
["sentence-transformers","combined",0.767957,0.043210,0.259283,0.373010,0.232043,70.491731,27.039215],

["distilbert","transformer_only",0.728395,0.009259,0.225229,0.365662,0.271605,10.495693,17.665909],
["distilbert","embedding",0.755892,0.040123,0.238714,0.373199,0.244108,9.662311,35.337234],
["distilbert","combined",0.705668,0.018519,0.260536,0.369212,0.294332,38.758854,53.563644],

["baseline","baseline",0.771605,0.049383,0.255041,0.384266,0.228395,7.184278,1.267029],
]

cols = [
    "Transformer",
    "Mode",
    "Accuracy",
    "Subset_Accuracy",
    "F1_Score",
    "F1_Micro",
    "Hamming_Loss",
    "Train_Time_sec",
    "Feature_Space_MB"
]

df = pd.DataFrame(data, columns=cols)

# ============================================================
# LABELS
# ============================================================

df["Label"] = df["Transformer"] + "\n" + df["Mode"]

# ============================================================
# SAVE CSV
# ============================================================

df.to_csv("all_model_results.csv", index=False)

# ============================================================
# COLORS
# ============================================================

palette = {
    "baseline": "#808080",
    "sentence-transformers": "#1f77b4",
    "distilbert": "#2ca02c",
    "roberta": "#d62728"
}

bar_colors = [palette[x] for x in df["Transformer"]]

handles = [
    plt.Rectangle((0,0),1,1,color=v,label=k)
    for k,v in palette.items()
]

# ============================================================
# PLOT 1 : ACCURACY
# ============================================================

plt.figure(figsize=(14,6))

bars = plt.bar(df["Label"], df["Accuracy"], color=bar_colors)

plt.title("Accuracy Comparison Across Models", fontsize=18, weight="bold")
plt.ylabel("Accuracy")
plt.xticks(rotation=40, ha="right")

for b in bars:
    h = b.get_height()
    plt.text(
        b.get_x() + b.get_width()/2,
        h + 0.003,
        f"{h:.3f}",
        ha="center",
        fontsize=9
    )

plt.legend(handles=handles, title="Transformer Family")
plt.tight_layout()
plt.savefig("plot1_accuracy.png", dpi=300)
plt.close()

# ============================================================
# PLOT 2 : F1 SCORE
# ============================================================

plt.figure(figsize=(14,6))

bars = plt.bar(df["Label"], df["F1_Score"], color=bar_colors)

plt.title("F1 Score Comparison Across Models", fontsize=18, weight="bold")
plt.ylabel("F1 Score")
plt.xticks(rotation=40, ha="right")

for b in bars:
    h = b.get_height()
    plt.text(
        b.get_x() + b.get_width()/2,
        h + 0.002,
        f"{h:.3f}",
        ha="center",
        fontsize=9
    )

plt.legend(handles=handles, title="Transformer Family")
plt.tight_layout()
plt.savefig("plot2_f1_score.png", dpi=300)
plt.close()

# ============================================================
# PLOT 3 : F1 vs SPACE
# ============================================================

plt.figure(figsize=(12,7))

for _, row in df.iterrows():
    plt.scatter(
        row["Feature_Space_MB"],
        row["F1_Score"],
        s=250,
        color=palette[row["Transformer"]],
        alpha=0.85
    )

    plt.text(
        row["Feature_Space_MB"] + 1,
        row["F1_Score"],
        row["Mode"],
        fontsize=10
    )

plt.title("Feature Space vs F1 Score", fontsize=18, weight="bold")
plt.xlabel("Feature Space (MB)")
plt.ylabel("F1 Score")
plt.legend(handles=handles, title="Transformer Family")
plt.tight_layout()
plt.savefig("plot3_f1_vs_space.png", dpi=300)
plt.close()

# ============================================================
# PLOT 4 : F1 vs TRAIN TIME
# ============================================================

plt.figure(figsize=(12,7))

for _, row in df.iterrows():
    plt.scatter(
        row["Train_Time_sec"],
        row["F1_Score"],
        s=250,
        color=palette[row["Transformer"]],
        alpha=0.85
    )

    plt.text(
        row["Train_Time_sec"] + 1,
        row["F1_Score"],
        row["Mode"],
        fontsize=10
    )

plt.title("Training Time vs F1 Score", fontsize=18, weight="bold")
plt.xlabel("Training Time (sec)")
plt.ylabel("F1 Score")
plt.legend(handles=handles, title="Transformer Family")
plt.tight_layout()
plt.savefig("plot4_f1_vs_time.png", dpi=300)
plt.close()

# ============================================================
# DONE
# ============================================================

print("Saved files:")
print("1. all_model_results.csv")
print("2. plot1_accuracy.png")
print("3. plot2_f1_score.png")
print("4. plot3_f1_vs_space.png")
print("5. plot4_f1_vs_time.png")