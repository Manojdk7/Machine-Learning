import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
)

# --------------------------------------------------
# 1. Load the heart disease dataset
# --------------------------------------------------
base_dir = Path(__file__).resolve().parent
data_path = base_dir / "heart.csv"

if data_path.exists():
    df = pd.read_csv(data_path)
else:
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.cleveland.data"
    columns = [
        "age", "sex", "cp", "trestbps", "chol", "fbs", "restecg",
        "thalach", "exang", "oldpeak", "slope", "ca", "thal", "target"
    ]
    df = pd.read_csv(url, header=None, names=columns)
    df = df.replace("?", pd.NA)
    for col in ["ca", "thal"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna().reset_index(drop=True)
    df.to_csv(data_path, index=False)

print("Dataset shape:", df.shape)
print(df.head())

# --------------------------------------------------
# 2. Prepare features and target
# --------------------------------------------------
X = df.drop(columns=["target"])
y = (df["target"] > 0).astype(int)

# --------------------------------------------------
# 3. Train/Test split
# --------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y,
)

# --------------------------------------------------
# 4. Function to evaluate a model
# --------------------------------------------------
def evaluate_model(criterion_name):
    model = DecisionTreeClassifier(criterion=criterion_name, random_state=42)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    cm = confusion_matrix(y_test, y_pred)

    print(f"\n=== {criterion_name.upper()} ===")
    print(f"Accuracy: {acc:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print("Confusion Matrix:")
    print(cm)
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, digits=4, target_names=["No Heart Disease", "Heart Disease"]))

    return {
        "criterion": criterion_name,
        "accuracy": acc,
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "confusion_matrix": cm,
    }

# --------------------------------------------------
# 5. Compare Gini vs Entropy
# --------------------------------------------------
gini_result = evaluate_model("gini")
entropy_result = evaluate_model("entropy")

print("\nComparison Summary:")
for result in [gini_result, entropy_result]:
    print(
        f"Criterion={result['criterion']}, "
        f"Accuracy={result['accuracy']:.4f}, "
        f"Precision={result['precision']:.4f}, "
        f"Recall={result['recall']:.4f}, "
        f"F1={result['f1_score']:.4f}"
    )

# --------------------------------------------------
# 6. Example prediction
# --------------------------------------------------
sample = pd.DataFrame([
    {
        "age": 52,
        "sex": 1,
        "cp": 0,
        "trestbps": 125,
        "chol": 212,
        "fbs": 0,
        "restecg": 1,
        "thalach": 168,
        "exang": 0,
        "oldpeak": 1.0,
        "slope": 2,
        "ca": 2,
        "thal": 3,
    }
])

best_model = DecisionTreeClassifier(criterion="gini", random_state=42)
best_model.fit(X_train, y_train)

# --------------------------------------------------
# 7. Save decision tree graph
# --------------------------------------------------
plt.figure(figsize=(22, 12))
plot_tree(
    best_model,
    feature_names=X.columns,
    class_names=["No Heart Disease", "Heart Disease"],
    filled=True,
    rounded=True,
    impurity=False,
)
plt.title("Heart Disease Decision Tree")
plt.tight_layout()
output_path = base_dir / "heart_disease_decision_tree.png"
plt.savefig(output_path, dpi=300)
print(f"\nDecision tree graph saved to: {output_path}")
plt.close()

prediction = best_model.predict(sample)[0]
print(f"\nPredicted class for sample: {prediction}")
if prediction == 1:
    print("Prediction: Heart disease present")
else:
    print("Prediction: No heart disease")