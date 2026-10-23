"""Decision Tree | test_size = 0.7 | CICIDS2017 error testing"""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (confusion_matrix, accuracy_score, precision_score,
                             recall_score, f1_score, classification_report)

ALGO = "Decision Tree"
TEST_SIZE = 0.7

HERE = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.environ.get("DATA_DIR", HERE)      # folder holding the CSV files
OUT_DIR = os.environ.get("OUT_DIR", HERE)        # folder for the jpg
OUT_FILE = os.path.join(OUT_DIR, os.path.splitext(os.path.basename(__file__))[0] + ".jpg")
FILES = [
    "Tuesday-WorkingHours.pcap_ISCX.csv",
    "Wednesday-workingHours.pcap_ISCX.csv",
    "Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv",
]

# ---- load and clean -------------------------------------------------------
dataset = pd.concat(
    [pd.read_csv(os.path.join(DATA_DIR, f), encoding="latin-1") for f in FILES],
    ignore_index=True,
)
dataset.columns = dataset.columns.str.strip()

X = dataset.iloc[:, :-1].apply(pd.to_numeric, errors="coerce")
X = X.replace([np.inf, -np.inf], np.nan)
le = LabelEncoder()
y = le.fit_transform(dataset.iloc[:, -1])

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=TEST_SIZE, random_state=0, stratify=y
)

# ---- model (the imputer is fitted on the training part only) --------------
model = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("clf", DecisionTreeClassifier(criterion="entropy", random_state=0)),
])
model.fit(X_train, y_train)

y_pred_train = model.predict(X_train)
y_pred = model.predict(X_test)

# ---- error testing --------------------------------------------------------
train_error = 1 - accuracy_score(y_train, y_pred_train)
test_error = 1 - accuracy_score(y_test, y_pred)
labels = np.arange(len(le.classes_))
cm = confusion_matrix(y_test, y_pred, labels=labels)

print(f"{ALGO} | test_size = {TEST_SIZE}")
print(f"Train rows: {len(y_train)} | Test rows: {len(y_test)}")
print(f"Train error : {train_error:.4f}")
print(f"Test error  : {test_error:.4f}")
print(f"Accuracy    : {1 - test_error:.4f}")
for avg in ("weighted", "macro"):
    print(f"Precision ({avg}): {precision_score(y_test, y_pred, average=avg, zero_division=0):.4f}")
    print(f"Recall    ({avg}): {recall_score(y_test, y_pred, average=avg, zero_division=0):.4f}")
    print(f"F1        ({avg}): {f1_score(y_test, y_pred, average=avg, zero_division=0):.4f}")
print("\nClassification Report:\n")
print(classification_report(y_test, y_pred, labels=labels,
                            target_names=le.classes_, zero_division=0))

# ---- plot -----------------------------------------------------------------
macro_f1 = f1_score(y_test, y_pred, average="macro", zero_division=0)
cm_norm = cm.astype(float) / np.maximum(cm.sum(axis=1, keepdims=True), 1)
short = [str(c)[:14] for c in le.classes_]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6.5), gridspec_kw={"width_ratios": [1, 2.2]})

bars = ax1.bar(["Train error", "Test error"], [train_error, test_error],
               color=["#4C72B0", "#DD8452"])
for b, v in zip(bars, [train_error, test_error]):
    ax1.text(b.get_x() + b.get_width() / 2, v, f"{v:.4f}", ha="center", va="bottom", fontsize=11)
ax1.set_ylim(0, max(train_error, test_error, 0.01) * 1.25)
ax1.set_ylabel("Error rate (1 - accuracy)")
ax1.set_title("Train vs test error")

im = ax2.imshow(cm_norm, cmap="Blues", vmin=0, vmax=1)
ax2.set_xticks(labels)
ax2.set_yticks(labels)
ax2.set_xticklabels(short, rotation=60, ha="right", fontsize=8)
ax2.set_yticklabels(short, fontsize=8)
for i in range(len(labels)):
    for j in range(len(labels)):
        if cm_norm[i, j] >= 0.005:
            ax2.text(j, i, f"{cm_norm[i, j]:.2f}", ha="center", va="center", fontsize=7,
                     color="white" if cm_norm[i, j] > 0.5 else "black")
ax2.set_xlabel("Predicted")
ax2.set_ylabel("Actual")
ax2.set_title("Confusion matrix (row-normalised)")
fig.colorbar(im, ax=ax2, fraction=0.046, pad=0.04)

fig.suptitle(f"{ALGO} | test_size = {TEST_SIZE} | test error = {test_error:.4f} | macro F1 = {macro_f1:.4f}",
             fontsize=13)
fig.tight_layout(rect=[0, 0, 1, 0.95])
fig.savefig(OUT_FILE, dpi=150, format="jpg")
print(f"\nSaved plot to {OUT_FILE}")
