import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve

# Load one horizon (example: h4)
df = pd.read_parquet(
    "modeling_pipeline_bin/models/final_model/outputs/h4/final_test_predictions.parquet"
)

y_true = df["actual_label"].to_numpy()
y_prob = df["prob_not_quiet"].to_numpy()

precision, recall, thresholds = precision_recall_curve(y_true, y_prob)

plt.figure(figsize=(6, 5))
plt.plot(recall, precision, linewidth=2)
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("Precision–Recall Curve (horizon = 4h)")
plt.grid(True)
plt.tight_layout()
plt.show()

